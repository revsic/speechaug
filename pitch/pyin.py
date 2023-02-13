from typing import Tuple

import numpy as np
import torch

from .yin import YIN, localmin, parabolic_interp


def viterbi(log_prob: torch.Tensor, log_trans: torch.Tensor, log_init: torch.Tensor) -> torch.Tensor:
    """Viterbi algorithm.
    Args:
        log_prob: [torch.float32; [..., S, bins]], conditional log-likelihood.
        log_trans: [torch.float32; [bins, bins]], transition log-probability.
        log_init: [torch.float32; [bins]], log-probability of initial state.
    Returns:
        states: [torch.long; [..., S]], index sequence.
    """
    # S
    steps = log_prob.shape[-2]
    # [..., bins]
    value = log_prob[..., 0, :] + log_init
    # [..., S, bins]
    ptrs = torch.zeros_like(log_prob)
    for i in range(1, steps):
        # [..., bins]
        max_val, ptrs[..., i, :] = (value[..., None] + log_trans).max(dim=-2)
        # [..., bins]
        value = log_prob[..., i, :] + max_val
    # [..., S], backtracking
    states = torch.zeros_like(log_prob[..., 0], dtype=torch.long)
    # initial state
    states[..., -1] = value.argmax(dim=-1)
    for i in range(steps - 2, -1, -1):
        # [...]
        states[..., i] = ptrs[..., i + 1, :].gather(
            -1, states[..., i + 1, None]).squeeze(dim=-1)
    return states

class pYIN(YIN):
    """pYIN-based pitch estimation algorithm.
    """
    def __init__(self,
                 sr: int,
                 frame_time: float = 0.01,
                 freq_min: float = 75.,
                 freq_max: float = 600.,
                 down_sr: int = 16000,
                 bins_per_octave: int = 12,
                 bins_per_semitone: int = 100,
                 beta_parameters: Tuple[int, float] = (2, 18.),
                 num_thresholds = 100,
                 switch_prob: float = 0.01,
                 lambda_: float = 2.,
                 max_octave_trans: float = 35.92,
                 no_trough_prob: float = 0.01):
        """Initializer.
        Args:
            sr: sampling rate.
            frame_time: duration of the frame.
            freq_min, freq_max: frequency min and max.
            down_sr: downsampling sr for fast computation.
            bins_per_octave: the number of the bins(semitones) in octave.
            bins_per_semitone: the number of the bins in semitone.
            beta_parameters: tuple of alpha, beta.
            num_thresholds: the number of the harmonicity thresholds.
            switch_prob: probability to swtich voiced to unvoiced.
            lambda_: lambda value of boltzmann distribution (truncated discrete exponential).
            max_octave_trans: maximum octave steps of transition per second.
            no_trough_prob: 
        """
        super().__init__(
            sr,
            frame_time,
            freq_min,
            freq_max,
            threshold=None,
            median_win=None,
            down_sr=down_sr)
        # pYIN parameters
        self.fmin = freq_min
        self.bins_per_octave = bins_per_octave
        self.bins_per_semitone = bins_per_semitone
        self.lambda_ = lambda_
        self.no_trough_prob = no_trough_prob
        self.register_pyin_state(
            frame_time,
            freq_min,
            freq_max,
            bins_per_octave,
            bins_per_semitone,
            beta_parameters,
            num_thresholds,
            switch_prob,
            max_octave_trans)

    def sample(self, cmnd: torch.Tensor) -> torch.Tensor:
        """Sample pitch frequency based on Viterbi-path searching.
        Args:
            cmnd: [torch.float32; [..., T / strides, tau_max - tau_min]],
                framed cumulative mean normalized difference.
        Returns:
            [torch.float32; [..., S]], pitch sequence.
        """
        device = cmnd.device
        # [..., T / strides, tau_max - tau_min]
        lmin = localmin(cmnd)
        # [..., T / strides, tau_max - tau_min, num_thresholds]
        tholds = lmin[..., None] & (cmnd[..., None] < self.thresholds[1:])
        # [..., T / strides, tau_max - tau_min, num_thresholds]
        positions = torch.cumsum(tholds, dim=-2)
        # boltzmann prior, truncated exponential
        k, N = positions - 1, positions[..., -1, :]
        _fact = np.expm1(-self.lambda_) / (-self.lambda_ * N).expm1()
        prior = _fact[..., None, :] * (-self.lambda_ * k).exp()
        # masking
        prior[~tholds] = 0.
        # [..., T / strides, tau_max - tau_min]
        probs = torch.matmul(prior, self.beta_probs[:, None]).squeeze(dim=-1)

        ## add prob to global minima if no candidates below the threshold
        ## else add prob to each candidates below the threshold
        # [..., T / strides]
        global_min = cmnd.masked_fill(~lmin, np.inf).argmin(dim=-1)
        # alias
        num_tholds = tholds.shape[-1]
        # [..., T / strides, 1, num_thresholds], threshold of global min
        holds = tholds.gather(
            -2,
            global_min[..., None, None].repeat(
                [1] * cmnd.dim() + [num_tholds]))
        # [..., T / strides]
        below_min = torch.count_nonzero(~holds.squeeze(dim=-2), dim=-1)
        # [tholds]
        a = torch.arange(num_tholds, device=device)
        # add probs
        probs.scatter_add_(
            -1,
            global_min[..., None],
            self.no_trough_prob * (
                (a < below_min[..., None]).float() * self.beta_probs).sum(dim=-1, keepdim=True))

        # [tau_max - tau_min]
        tau = torch.arange(self.tau_max - self.tau_min, device=device)
        # [..., T / strides, tau_max - tau_min]
        pshifts = parabolic_interp(cmnd)
        # refining peak
        period = tau + self.tau_min + 1 + pshifts
        # [..., T / strides, tau_max - tau_min], to frequency
        f0s = self.down_sr / period.clamp_min(1e-5)
        # [..., T / strides, tau_max - tau_min], quantize
        bins = self.bins_per_octave * self.bins_per_semitone * (f0s / self.fmin).log2()
        bins = bins.round().clamp(0, self.total_bins - 1).long()

        # [..., T / strides, 2 x total_bins], observation probs
        observed = torch.zeros(*bins.shape[:-1], self.total_bins * 2, device=device)
        observed.scatter_(-1, bins, probs)
        # voice probs
        voiced = observed[..., :self.total_bins].sum(dim=-1).clamp(0., 1.)
        observed[..., self.total_bins:] = (1 - voiced[..., None]) / self.total_bins

        # path search
        # [..., T / strides]
        states = viterbi(
            observed.clamp_min(1e-7).log(),
            self.transition.clamp_min(1e-7).log(),
            self.p_init.clamp_min(1e-7).log())
        # convert to frequency
        f0 = self.freqs[states % self.total_bins]
        # if in voice
        voiced_flag = states < self.total_bins
        # unvoice to zero
        f0[~voiced_flag] = 0.
        return f0

    def register_pyin_state(self,
                            frame_time: float,
                            fmin: float,
                            fmax: float,
                            bins_per_octave: int,
                            bins_per_semitone: int,
                            beta_parameters: Tuple[int, float],
                            num_thresholds: int,
                            switch_prob: float,
                            max_octave_trans: float):
        # [num_thresholds + 1]
        self.register_buffer(
            'thresholds',
            torch.linspace(0., 1., num_thresholds + 1),
            persistent=False)
        # beta-distribution prior
        import scipy.stats
        a, b = beta_parameters
        beta_cdf = torch.tensor(scipy.stats.beta.cdf(self.thresholds, a, b), dtype=torch.float32)
        # [num_thresholds]
        self.register_buffer(
            'beta_probs',
            beta_cdf.diff(),
            persistent=False)

        # the number of the possible bins
        total_bins = int(
            bins_per_octave * bins_per_semitone * np.log2(fmax / fmin))
        self.total_bins = total_bins
        # maximum octave steps of transition per frame
        max_semitones_per_frame = round(
            max_octave_trans * bins_per_octave * frame_time)
        # local transition matrix with triangular window
        ## transition[i, j] = 0 if |i - j| > width
        ## transition[i, i] is maximal
        ## transition[i, i - width // 2:i + width // 2] = window
        w = max_semitones_per_frame * bins_per_semitone + 1
        # [total_bins]
        a = torch.arange(total_bins)
        # [total_bins, total_bins], bin-transition probs
        grid = (w // 2 + 1 - (a - a[:, None]).abs()).clamp_min(0)
        transition = grid / grid.sum(dim=-1)

        # self-loop transition matrix
        ## transition[i, i] = p for all i
        ## transition[i, j] = (1 - p) / (states - 1) for all i != j
        # [2, 2], voice-transition probs
        t_switch = torch.full((2, 2), switch_prob)
        t_switch[0, 0] = 1 - switch_prob
        t_switch[1, 1] = 1 - switch_prob
        # [2 x total_bins, 2 x total_bins], apply voice-probs
        self.register_buffer(
            'transition',
            torch.kron(t_switch, transition),
            persistent=False)

        # [2 x total_bins], initial probs, unvoiced
        p_init = torch.zeros(2 * total_bins)
        p_init[total_bins:] = 1 / total_bins
        self.register_buffer('p_init', p_init, persistent=False)

        # [total_bins]
        self.register_buffer(
            'freqs',
            fmin * 2 ** (a / (bins_per_octave * bins_per_semitone)),
            persistent=False)
