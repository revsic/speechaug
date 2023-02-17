"""
Copyright (C) https://github.com/praat/praat

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PitchShift(nn.Module):
    """Pitch shift with formant correction, based on TD-PSOLA
    """
    def __init__(self,
                 sr: int,
                 floor: float = 60):
        """Initializer.
        Args:
            sr: sampling rate.
            floor: floor value of fundamental frequency.
        """
        super().__init__()
        self.sr = sr
        self.floor = floor

    def forward(self,
                snd: torch.Tensor,
                pitch: torch.Tensor,
                pitch_shift: float,
                pitch_range: float) -> torch.Tensor:
        """Shift the pitch of given speech tensor.
        Args:
            snd: [torch.float32; [T]], mono-channel, [-1, 1]-ranged.
            pitch: [torch.float32; [S]], pitch sequence, hertz-level.
            pitch_shift: pitch shifting factor.
            pitch_range: pitch ranging factor.
        Returns:
            [torch.float32; [T]], shifted.
        """
        # [T], mean normalization
        snd = snd - snd.mean()
        # [T]
        f0 = F.interpolate(pitch[None, None], size=len(snd), mode='linear')[0, 0]
        # [P], find all peaks in voiced segment
        peaks = self.find_allpeaks(snd, f0)

        # nonzero median
        median = f0[f0 > 0.].median().item() * pitch_shift
        # shift
        f0 = f0 * pitch_shift
        # rerange
        f0 = torch.where(
            f0 > 0.,
            median + (f0 - median) * pitch_range,
            0.)

        return self.psola(snd, f0, peaks)

    def find_voiced_segment(self, f0: torch.Tensor, i: int = 0) -> Optional[Tuple[int, int]]:
        """Find voiced segment starting from `i`.
        Args:
            f0: [torch.float32; [T]], fundamental frequencies, hertz-level.
        Returns:
            segment left and right if voiced segment exist (half inclusive range)
        """
        # if f0 tensor is empty
        if len(f0[i:]) == 0:
            return None
        # next voiced interval
        flag = (f0[i:] > 0.).long()
        # force first label if False
        flag[0] = 0
        # if all unvoiced
        if not flag.any():
            return None
        # if found
        left = i + flag.argmax()
        # count the numbers
        _, (_, cnt, *_) = flag.unique_consecutive(return_counts=True)
        right = left + cnt
        return left.item(), right.item()

    def find_allpeaks(self, signal: torch.Tensor, f0: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute all periods from signal.
        Args:
            signal: [torch.float32; [T]], speech signal, [-1, 1]-ranged.
            f0: [torch.float32; [T]], fundamental frequencies, hertz-level.
        Returns:
            [torch.long; [P]], peaks.
        """
        # []
        global_peak = signal.abs().max()
        def find_peak(i: int, dir_: Union['left', 'right']):
            # find peaks
            w = int(self.sr / f0[i].clamp_min(self.floor))
            s = max(i - w // 2, 0)
            if dir_ == 'left':
                # -1.25 - 0.5, -0.8 - 0.5
                cand_l, cand_r = max(int(i - 1.75 * w), 0), max(int(i - 1.3 * w), 0)
            if dir_ == 'right':
                # 0.8 - 0.5, 1.25 - 0.5
                cand_l, cand_r = int(i + 0.3 * w), int(i + 0.75 * w)
            # if nothing to find
            if cand_l == cand_r or len(signal) - cand_l < w:
                return w, -1, i, 0
            # [cand(=cand_r - cand_l), w]
            seg = signal[cand_l:cand_r + w].unfold(-1, w, 1)
            # [cand]
            corr = torch.matmul(
                F.normalize(seg, dim=-1),
                F.normalize(signal[s:s + w], dim=-1)[:, None]).squeeze(dim=-1)
            # []
            max_corr, r = corr.max(), corr.argmax()
            peak = seg[r].abs().max()
            # add bias (cand_l - s)
            return w, max_corr, i + (r + cand_l) - s, peak

        added_right, i, peaks = -1e308, 0, []
        while True:
            voiced = self.find_voiced_segment(f0, i)
            if voiced is None:
                break
            # if exist
            left, right = voiced

            # middle interval
            middle = (left + right) // 2

            # find first extremum
            w = int(self.sr / f0[middle])
            s = max(middle - w // 2, 0)
            # []
            minima, imin = signal[s:s + w].min(), signal[s:s + w].argmin()
            maxima, imax = signal[s:s + w].max(), signal[s:s + w].argmax()
            # if all same
            if minima == maxima:
                i = middle
            else:
                i = s + (imin if abs(minima) > abs(maxima) else imax)

            backup = i
            # left interval search
            while True:
                w, corr, i, peak = find_peak(i, 'left')
                if corr == -1.:
                    i -= w
                if i < left:
                    if corr > 0.7 and peak > 0.023333 * global_peak and i - added_right > 0.8 * w:
                        peaks.append(i)
                    break
                if corr > 0.3 and (peak == 0. or peak > 0.01 * global_peak):
                    if i - added_right > 0.8 * w:
                        peaks.append(i)
            i = backup
            # right interval search
            while True:
                w, corr, i, peak = find_peak(i, 'right')
                if corr == -1.:
                    i += w
                # half-exclusive
                if i >= right:
                    if corr > 0.7 and peak > 0.023333 * global_peak:
                        peaks.append(i)
                        added_right = i
                    break
                if corr > 0.3 and (peak == 0. or peak > 0.01 * global_peak):
                    peaks.append(i)
                    added_right = i
            # to next interval
            i = right
        if len(peaks) == 0:
            return None
        # sort the point
        return torch.stack(sorted(peaks)).clamp(0, len(signal) - 1)

    def psola(self, signal: torch.Tensor, pitch: torch.Tensor, peaks: torch.Tensor) -> torch.Tensor:
        """Pitch-synchronous overlap and add.
        Args:
            signal: [torch.float32; [T]], speech signal, [-1, 1]-ranged.
            pitch: [torch.float32; [T]], fundamental frequencies, hertz-level.
            peaks: [torch.float32; [P]], peaks.
        Returns:
            [torch.float32; [T]], resampled.
        """
        device = signal.device
        max_w = 1.25 * self.sr / pitch[pitch > 0].min()
        # [T]
        new_signal = torch.zeros_like(signal)

        i = 0
        while i < len(signal):
            voiced = self.find_voiced_segment(pitch, i)
            if voiced is None:
                break
            # if voice found
            left_v, right_v = voiced
            # copy noise
            window = torch.hann_window(left_v - i, device=device)
            new_signal[i:left_v] += window * signal[i:left_v]

            while left_v < right_v:
                # find nearest peak
                p = (peaks - left_v).abs().argmin()
                period = int(self.sr / pitch[left_v].clamp_min(self.floor))
                # width
                left_w, right_w = period // 2, period // 2
                # clamping
                if p > 0 and peaks[p] - peaks[p - 1] <= max_w:
                    left_w = min(peaks[p] - peaks[p - 1], left_w)
                if p < len(peaks) - 1 and peaks[p + 1] - peaks[p] <= max_w:
                    right_w = min(peaks[p + 1] - peaks[p], right_w)
                # offset to index
                left_i, right_i = max(peaks[p] - left_w, 0), peaks[p] + right_w
                # copy
                ival = (right_i - left_i) // 2
                window = torch.hann_window(ival * 2, device=device)
                seglen = min(
                    len(new_signal[left_v - ival:left_v + ival]),
                    len(signal[left_i:left_i + ival * 2]))
                new_signal[left_v - ival:left_v - ival + seglen] += \
                    (window[:seglen] * signal[left_i:left_i + seglen])
                # next
                left_v += ival * 2
            # next segment
            i = right_v
        # copy last noise
        new_signal[i:] = signal[i:]
        return new_signal
