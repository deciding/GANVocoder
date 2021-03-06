# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from parallel_wavegan.layers import CausalConv1d
from parallel_wavegan.layers import CausalConvTranspose1d
from parallel_wavegan.layers import ResidualStack
from parallel_wavegan.layers import ResidualAdvancedStack

class MelGANAdvancedGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 kernel_size=7,
                 channels=512,
                 bias=True,
                 upsample_scales=[8, 8, 2, 2],
                 stack_kernel_size=3,
                 stacks=3,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_final_nonlinear_activation=True,
                 use_weight_norm=True,
                 use_causal_conv=False,
                 use_senet=False,
                 use_1x1skip=True,
                 use_multi_receptive_fusion=False,
                 mrf_kernels=[3, 7, 11, 13],
                 mrf_inner_dilations=[1, 3, 5],
                 ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANAdvancedGenerator, self).__init__()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert stacks == len(mrf_kernels)
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        self.use_multi_receptive_fusion = use_multi_receptive_fusion

        # add initial layer
        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(in_channels, channels, kernel_size,
                             bias=bias, pad=pad, pad_params=pad_params),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            # add residual stack
            for j in range(stacks):
                if use_multi_receptive_fusion:
                    stack_kernel_size = mrf_kernels[j]
                    dilation = mrf_inner_dilations
                else:
                    dilation = stack_kernel_size ** j
                layers += [
                    ResidualAdvancedStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=dilation,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                        use_senet = use_senet,
                        use_1x1skip = use_1x1skip,
                    )
                ]

        # add final layer
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(channels // (2 ** (i + 1)), out_channels, kernel_size,
                             bias=bias, pad=pad, pad_params=pad_params),
            ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        # initialize pqmf for inference
        self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        return self.melgan(c)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def inference(self, c):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)

class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 kernel_size=7,
                 channels=512,
                 bias=True,
                 upsample_scales=[8, 8, 2, 2],
                 stack_kernel_size=3,
                 stacks=3,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_final_nonlinear_activation=True,
                 use_weight_norm=True,
                 use_causal_conv=False,
                 ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANGenerator, self).__init__()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        # add initial layer
        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(in_channels, channels, kernel_size,
                             bias=bias, pad=pad, pad_params=pad_params),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        # add final layer
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(channels // (2 ** (i + 1)), out_channels, kernel_size,
                             bias=bias, pad=pad, pad_params=pad_params),
            ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        # initialize pqmf for inference
        self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        return self.melgan(c)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def inference(self, c):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)


class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 ):
        """Initilize MelGAN discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.

        """
        super(MelGANDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs, out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs, out_chs, kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs, out_channels, kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of output tensors of each layer.

        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs

class MelGANPeriodDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(self,
                 period,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad2d",
                 pad_params={},
                 fixed_kernel_size=False,
                 num_channels=None,
                 num_groups=None,
                 use_cond=False,
                 dim_spk=0,
                 ):
        """Initilize MelGAN discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.

        """
        super(MelGANPeriodDiscriminator, self).__init__()
        self.period = period
        self.layers = torch.nn.ModuleList()
        self.use_cond = use_cond
        if not use_cond:
            dim_spk = 0
        self.dim_spk = dim_spk

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1
        if num_channels is not None:
            assert len(downsample_scales)==len(num_channels)
        if num_groups is not None:
            assert len(downsample_scales)==len(num_groups)

        # add first layer
        # TODO: hifigan doesn't have the conv_in
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((0, 0, (np.prod(kernel_sizes) - 1) // 2, (np.prod(kernel_sizes) - 1) // 2), **pad_params),
                torch.nn.Conv2d(in_channels, channels, (np.prod(kernel_sizes), 1), bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        self.start_downsample=len(self.layers)
        self.end_downsample=self.start_downsample+len(downsample_scales)
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels) if num_channels is None else num_channels[i]
            group = in_chs // 4 if num_groups is None else num_groups[i]
            group = 1 if fixed_kernel_size else group
            kk = 5 if fixed_kernel_size else downsample_scale * 10 + 1
            ss = 3 if fixed_kernel_size else downsample_scale
            pp = 2 if fixed_kernel_size else downsample_scale * 5
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs + dim_spk, out_chs,
                        kernel_size=(kk, 1),
                        stride=(ss, 1),
                        padding=(pp, 0),
                        groups=group,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs

        # add final layers
        #out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_chs, out_chs, (kernel_sizes[0], 1),
                    padding=((kernel_sizes[0] - 1) // 2, 0),
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv2d(
                out_chs, out_channels, (kernel_sizes[1], 1),
                padding=((kernel_sizes[1] - 1) // 2, 0),
                bias=bias,
            ),
        ]

    def forward(self, x, c=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Input condition (B, C, 1, 1).

        Returns:
            List: List of output tensors of each layer.

        """
        #assert (c is None)!=self.use_cond
        if c is not None:
            assert c.shape[1]==self.dim_spk
        b, ch, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, ch, t // self.period, self.period)
        outs = []
        for i, f in enumerate(self.layers):
            if self.use_cond and i>=self.start_downsample and i<self.end_downsample:
                x = torch.cat([x, c.repeat(1, 1, x.size(2), x.size(3))], dim=1)
            x = f(x)
            out = torch.flatten(x, 1, -1)
            outs += [out]

        return outs

class MelGANCondDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 num_channels=None,
                 num_groups=None,
                 use_cond=False,
                 #num_spk=104,
                 dim_spk=32,
                 ):
        """Initilize MelGAN discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.

        """
        super(MelGANCondDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.use_cond = use_cond
        if not use_cond:
            dim_spk = 0
        self.dim_spk = dim_spk

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        ##TODO:delete
        #if use_cond:
        #    self.spk_embedding=torch.nn.Embedding(num_spk, dim_spk)
        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        self.start_downsample=len(self.layers)
        self.end_downsample=self.start_downsample+len(downsample_scales)
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels) if num_channels is None else num_channels[i]
            group = in_chs // 4 if num_groups is None else num_groups[i]
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs + dim_spk, out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=group,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs

        # add final layers
        #out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs, out_chs, kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs, out_channels, kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x, c=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Input condition (B, C, 1).

        Returns:
            List: List of output tensors of each layer.

        """
        #assert (c is None)!=self.use_cond
        if c is not None:
            assert c.shape[1]==self.dim_spk
        outs = []
        ##TODO:delete
        #if self.use_cond:
        #    spk_emd = self.spk_embedding(spk_idx).unsqueeze(2) # B, N, 1
        for i, f in enumerate(self.layers):
            if self.use_cond and i>=self.start_downsample and i<self.end_downsample:
                x = torch.cat([x, c.repeat(1, 1, x.size(2))], dim=1)
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 scales=3,
                 downsample_pooling="AvgPool1d",
                 # follow the official implementation setting
                 downsample_pooling_params={
                     "kernel_size": 4,
                     "stride": 2,
                     "padding": 1,
                     "count_include_pad": False,
                 },
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_weight_norm=True,
                 ):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(**downsample_pooling_params)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

class MelGANMultiScaleCondDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 scales=3,
                 downsample_pooling="AvgPool1d",
                 # follow the official implementation setting
                 downsample_pooling_params={
                     "kernel_size": 4,
                     "stride": 2,
                     "padding": 1,
                     "count_include_pad": False,
                 },
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_weight_norm=True,
                 num_channels=None,
                 num_groups=None,
                 use_cond=False,
                 num_spk=104,
                 dim_spk=32,
                 ):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANMultiScaleCondDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        self.use_cond = use_cond
        self.num_spk = num_spk

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANCondDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                    num_channels=num_channels,
                    num_groups=num_groups,
                    use_cond=use_cond,
                    #num_spk=num_spk,
                    dim_spk=dim_spk,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(**downsample_pooling_params)
        #TODO:fix
        if use_cond:
            self.spk_embedding=torch.nn.Embedding(num_spk, dim_spk)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x, spk_idx=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            spk_idx (Tensor): Input speaker one-hot label (B, C).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        #assert (spk_idx is None)!=self.use_cond
        #assert spk_idx.shape[1]==self.num_spk
        outs = []
        #TODO:fix
        if self.use_cond:
            spk_emd = self.spk_embedding(spk_idx).unsqueeze(2) # B, N, 1
        for f in self.discriminators:
            if self.use_cond:
                outs += [f(x, spk_emd)]
            else:
                outs += [f(x)]
            #outs += [f(x, spk_idx)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

class MelGANMultiPeriodCondDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 scales=3,
                 kernel_sizes=[5, 3],
                 channels=16,
                 max_downsample_channels=1024,
                 bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad2d",
                 pad_params={},
                 use_weight_norm=True,
                 num_channels=None,
                 num_groups=None,
                 use_cond=False,
                 num_spk=104,
                 dim_spk=32,
                 fixed_kernel_size=True,
                 periods=[2, 3, 5, 7, 11],
                 ):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANMultiPeriodCondDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        self.use_cond = use_cond
        self.num_spk = num_spk

        # add discriminators
        for period in periods:
            self.discriminators += [
                MelGANPeriodDiscriminator(
                    period=period,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                    fixed_kernel_size=fixed_kernel_size,
                    num_channels=num_channels,
                    num_groups=num_groups,
                    use_cond=use_cond,
                    dim_spk=dim_spk,
                )
            ]
        #TODO:fix
        if use_cond:
            self.spk_embedding=torch.nn.Embedding(num_spk, dim_spk)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x, spk_idx):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        #assert (spk_idx is None)!=self.use_cond
        #assert spk_idx.shape[1]==self.num_spk
        outs = []
        #TODO:fix
        if self.use_cond:
            spk_emd = self.spk_embedding(spk_idx).unsqueeze(2).unsqueeze(3) # B, N, 1
        for f in self.discriminators:
            if self.use_cond:
                outs += [f(x, spk_emd)]
            else:
                outs += [f(x)]

        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
