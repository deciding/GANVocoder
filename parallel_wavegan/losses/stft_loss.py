# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F
import librosa

from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

def complex_stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return real.transpose(2, 1), imag.transpose(2, 1)

class MagPhaseLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(MagPhaseLoss, self).__init__()

    def forward(self, x_real, x_imag, y_real, y_imag, x_mag=None, y_mag=None):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        if x_mag is None:
            x_mag = torch.sqrt(torch.clamp(x_real ** 2 + x_imag ** 2, min=1e-7))
        if y_mag is None:
            y_mag = torch.sqrt(torch.clamp(y_real ** 2 + y_imag ** 2, min=1e-7))

        return torch.mean( (x_mag - y_mag) ** 2 + 2 * (x_mag * y_mag - x_real * y_real - x_imag * y_imag) )

class WeightedPhaseLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(WeightedPhaseLoss, self).__init__()

    def forward(self, x_real, x_imag, y_real, y_imag, x_mag=None, y_mag=None):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        if x_mag is None:
            x_mag = torch.sqrt(torch.clamp(x_real ** 2 + x_imag ** 2, min=1e-7))
        if y_mag is None:
            y_mag = torch.sqrt(torch.clamp(y_real ** 2 + y_imag ** 2, min=1e-7))

        return torch.mean( (x_mag * y_mag - x_real * y_real - x_imag * y_imag) )

class PhaseLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(PhaseLoss, self).__init__()

    def forward(self, x_real, x_imag, y_real, y_imag, x_mag=None, y_mag=None):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        if x_mag is None:
            x_mag = torch.sqrt(torch.clamp(x_real ** 2 + x_imag ** 2, min=1e-7))
        if y_mag is None:
            y_mag = torch.sqrt(torch.clamp(y_real ** 2 + y_imag ** 2, min=1e-7))

        return torch.mean( 1 - (x_real * y_real + x_imag * y_imag) / (x_mag * y_mag + 1e-7) )

class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

def build_mel_basis(n_fft, sr=24000, n_mels=80):
    return torch.FloatTensor(librosa.filters.mel(sr, n_fft, n_mels=n_mels)).transpose(0, 1)

class LogMelLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self, fft_size, sr=24000, n_mels=80):
        """Initilize los STFT magnitude loss module."""
        super(LogMelLoss, self).__init__()
        self.register_buffer('mel_basis', build_mel_basis(fft_size, sr, n_mels))

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        x_mel = torch.matmul(x_mag, self.mel_basis.to(device=x_mag.device))
        y_mel = torch.matmul(y_mag, self.mel_basis.to(device=y_mag.device))
        x_logmel = torch.log(torch.clamp(x_mel, min=1e-5))
        y_logmel = torch.log(torch.clamp(y_mel, min=1e-5))
        return F.l1_loss(x_logmel, y_logmel)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss

class PhaseSTFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", queries=['sc', 'mag']):
        """Initialize STFT loss module."""
        super(PhaseSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.mag_phase_loss = MagPhaseLoss()
        self.weighted_phase_loss = WeightedPhaseLoss()
        self.phase_loss = PhaseLoss()
        self.log_mel_loss = LogMelLoss(fft_size)
        self.queries = queries


    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_real, x_imag = complex_stft(x, self.fft_size, self.shift_size, self.win_length, self.window.to(device=x.device))
        y_real, y_imag = complex_stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(device=x.device))
        x_mag = torch.sqrt(torch.clamp(x_real ** 2 + x_imag ** 2, min=1e-7))
        y_mag = torch.sqrt(torch.clamp(y_real ** 2 + y_imag ** 2, min=1e-7))

        res_map={}
        if 'sc' in self.queries:
            sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
            res_map['sc']=sc_loss
        if 'mag' in self.queries:
            mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
            res_map['mag']=mag_loss
        if 'mp' in self.queries:
            mp_loss = self.mag_phase_loss(x_real, x_imag, y_real, y_imag, x_mag, y_mag)
            res_map['mp']=mp_loss
        if 'wp' in self.queries:
            wp_loss = self.weighted_phase_loss(x_real, x_imag, y_real, y_imag, x_mag, y_mag)
            res_map['wp']=wp_loss
        if 'ph' in self.queries:
            ph_loss = self.phase_loss(x_real, x_imag, y_real, y_imag, x_mag, y_mag)
            res_map['ph']=ph_loss
        if 'mel' in self.queries:
            mel_loss = self.log_mel_loss(x_mag, y_mag)
            res_map['mel']=mel_loss

        return res_map

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss

class MultiResolutionSTFTLoss2(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window",
                 queries=['sc', 'mag', 'mel']):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss2, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.queries = queries
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [PhaseSTFTLoss(fs, ss, wl, window, queries)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        res_map={k:0.0 for k in self.queries}
        for f in self.stft_losses:
            sub_res_map = f(x, y)
            for k in sub_res_map:
                res_map[k]+=sub_res_map[k]
        for k in res_map:
            res_map[k] /= len(self.stft_losses)

        return res_map
