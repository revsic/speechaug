from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF


def localmin(x: torch.Tensor) -> torch.Tensor:
    """Local minima.
    Args:
        x: [torch.float32; [..., T]], input tensor.
    Returns:
        [bool; [..., T]], local minima.
    """
    # [..., T + 1], x[i + 1] - x[i]
    d = F.pad(x.diff(dim=-1), [1, 1])
    # [..., T], dec & inc
    return (d[..., :-1] <= 0) & (d[..., 1:] >= 0)


def parabolic_interp(x: torch.Tensor) -> torch.Tensor:
    """Parabolic interpolation for smoothing difference function.
    Args:
        x: [torch.float32; [..., C]], input tensor.
    Returns:
        [torch.float32; [..., C]], parabolic shifts.
    """
    # [..., C - 2], previous, current, next
    p, c, n = x[..., :-2], x[..., 1:-1], x[..., 2:]
    # assume x is convex, then a > 0
    a = n + p - 2 * c
    b = 0.5 * (n - p)
    # [..., C - 2]
    shifts = -b / a
    shifts[b.abs() >= a.abs()] = 0.
    # [..., C], edge
    return F.pad(shifts, [1, 1])


class PitchTracker(nn.Module):
    """YIN-based pitch estimation algorithm.
    """
    def __init__(self,
                 sr: int,
                 frame_time: float = 0.01,
                 freq_min: float = 75.,
                 freq_max: float = 600.,
                 threshold: float = 0.2,
                 median_win: Optional[int] = 3,
                 down_sr: int = 16000):
        """Initializer.
        Args:
            sr: sampling rate.
            frame_time: duration of the frame.
            freq_min, freq_max: frequency min and max.
            threshold: harmonicity threshold.
            median_win: length of the window for median smoothing.
            down_sr: downsampling sr for fast computation.
        """
        super().__init__()
        self.sr = sr
        self.strides = int(sr * frame_time)
        self.tau_max = int(down_sr // freq_min)
        self.tau_min = int(down_sr // freq_max)
        self.threshold = threshold
        self.median_win = median_win
        self.down_sr = down_sr

    @classmethod
    def cmnd(cls, signal: torch.Tensor, tmax: int, tmin: int) -> torch.Tensor:
        """Cumulative mean normalized difference.
        Args:
            signal: [torch.float32; [..., W]], input signal.
            tmax, tmin: maximum, minimum value of the time-lag.
        Returns:
            [torch.float32; [..., tmax - tmin]], CMND.
        """
        # one-based
        # d[tau]
        # = sum_{j=1}^{W-tau} (x[j] - x[j + tau])^2
        # = sum_{j=1}^{W-tau} (x[j]^2 - 2x[j]x[j + tau] + x[j + tau]^2)
        # = c[W - tau] - 2 * a[tau] + (c[W] - c[tau])
        #     where c[k] = sum_{j=1}^k x[j]^2
        #           a[tau] = sum_{j=1}^W x[j]x[j + tau]

        # W
        w = signal.shape[-1]
        # [..., W + 1]
        fft = torch.fft.rfft(signal, w * 2, dim=-1)
        # [..., W x 2], symmetric
        corr = torch.fft.irfft(fft * fft.conj(), dim=-1)
        # [..., W]
        cumsum = signal.square().cumsum(dim=-1)
        # [..., tmax], difference
        diff = (
            # c[W - tau]
            torch.flip(cumsum[..., -tmax:], dims=(-1,))
            # -2 x a[tau]
            - 2 * corr[..., :tmax]
            # + (c[W] - c[tau])
            + cumsum[..., -1, None] - cumsum[..., :tmax])
        # [..., tmax - 1], remove redundant
        cumdiff = diff[..., 1:] / (diff[..., 1:].cumsum(dim=-1) + 1e-7)
        # normalize
        cumdiff = cumdiff * torch.arange(1, tmax, device=cumdiff.device)
        # [..., tmax - tmin]
        return F.pad(cumdiff, [1, 0], value=1.)[..., tmin:]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Estimate the pitch from the input signal based on YIN.
        Args:
            inputs: [torch.float32; [..., T]], input audio.
        Returns:
            [torch.float32; [..., S]], pitch frequency.
        """
        down = AF.resample(inputs, self.sr, self.down_sr)
        # set windows based on tau-max
        w = int(2 ** np.ceil(np.log2(self.tau_max))) + 1
        # [..., T / strides, windows]
        frames = F.pad(down, [0, w]).unfold(-1, w, self.strides)
        # [..., T / strides, tau_max - tau_min], cumulative mean normalized difference
        cmnd = PitchTracker.cmnd(frames, self.tau_max, self.tau_min)
        # sampling
        return self.sample_yin(cmnd)

    def sample_yin(self, cmnd: torch.Tensor) -> torch.Tensor:
        """Sample pitch frequency locally.
        Args:
            cmnd: [torch.float32; [..., T / strides, tau_max - tau_min]],
                framed cumulative mean normalized difference.
        Returns:
            [torch.float32; [..., S]], pitch sequence.
        """
        # [..., T / strides]
        thold = (cmnd < self.threshold).long().argmax(dim=-1)
        # if not found
        thold[thold == 0] = self.tau_max - self.tau_min
        # [..., T / strides, tau_max - tau_min] switch mask
        thold = thold[..., None] <= torch.arange(
            self.tau_max - self.tau_min, device=cmnd.device)

        # [..., T / strides, tau_max - tau_min]
        lmin = localmin(cmnd)
        # [..., T / strides]
        tau = (thold & lmin).long().argmax(dim=-1)
        # if not found
        tau[tau == self.tau_max - self.tau_min - 1] == 0

        # [..., T / strides, tau_max - tau_min]
        pshifts = parabolic_interp(cmnd)
        # refining peak
        period = tau + self.tau_min + 1 + pshifts.gather(-1, tau[..., None]).squeeze(dim=-1)
        # [..., T / strides], to frequency
        pitch = torch.where(
            tau > 0,
            self.down_sr / period,
            torch.tensor(0., device=tau.device))
        # median pool
        if self.median_win is not None:
            pitch = torch.median(
                pitch.unfold(-1, self.median_win, 1),
                dim=-1).values
        return pitch
