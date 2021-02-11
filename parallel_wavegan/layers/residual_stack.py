# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Residual stack module in MelGAN."""

import torch
import torch.nn.functional as F

from parallel_wavegan.layers import CausalConv1d, CausalConv1dNoPad


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(self,
                 kernel_size=3,
                 channels=32,
                 dilation=1,
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_causal_conv=False,
                 ):
        """Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(ResidualStack, self).__init__()

        # defile residual stack part
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params),
                torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )
        else:
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                CausalConv1d(channels, channels, kernel_size, dilation=dilation,
                             bias=bias, pad=pad, pad_params=pad_params),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, chennels, T).

        """
        return self.stack(c) + self.skip_layer(c)

# Squeeze and Excitation Block Module
class SEBlock(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels // reduction, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1) # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation
        w = torch.sigmoid(w)

        return x * w + b # Scale and add bias

class ResidualAdvancedStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(self,
                 kernel_size=3,
                 channels=32,
                 dilation=1,
                 #dilation=[1, 3, 5],
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_causal_conv=False,
                 reduction=16,
                 use_senet=False,
                 use_1x1skip=True,
                 ):
        """Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(ResidualAdvancedStack, self).__init__()
        self.use_senet = use_senet
        self.use_1x1skip = use_1x1skip
        use_multi_dilations = isinstance(dilation, list)
        self.use_multi_dilations = use_multi_dilations

        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            pad_kernel=(kernel_size-1)//2*dilation
            conv_type=torch.nn.Conv1d
        #self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        else:
            pad_kernel=(kernel_size-1)*dilation
            conv_type=CausalConv1dNoPad
        # defile residual stack part
        stack_layers=[]
        if use_multi_dilations:
            dilations=dilation
        else:
            dilations=[dilation]
        for dilation in dilations:
            stack_layers+=[
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                getattr(torch.nn, pad)(pad_kernel, **pad_params),
                conv_type(channels, channels, kernel_size, dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            ]
        self.stack = torch.nn.Sequential(*stack_layers)

        # defile extra layer for skip connection
        if use_1x1skip:
            self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)
        if use_senet:
            self.se_block = SEBlock(channels, reduction=reduction)

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, chennels, T).

        """
        org = c
        residual = self.stack(c)
        if self.use_senet:
            residual = self.se_block(residual)
        if self.use_1x1skip:
            org = self.skip_layer(c)
        #return self.stack(c) + self.skip_layer(c)
        #return self.se_block(self.stack(c)) + c
        return residual + org
