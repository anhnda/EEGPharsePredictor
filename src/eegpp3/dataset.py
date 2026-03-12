from typing import Literal

import joblib
import torch
from torch.utils.data import Dataset

from . import params
from .utils.data_utils import LABEL_DICT
from .utils.augmentation import SamplingRateDegradation


class EEGDataset(Dataset):
    def __init__(self, dump_path: str, w_out=3, contain_side: Literal['left', 'right', 'both', 'none'] = 'both',
                 is_infer=False, minmax_normalized=True, enable_degradation=False, degradation_prob=0.3):
        """
        EEG Dataset
        :param dump_path:
        :param w_out: must be an odd if contain == 'both'
        :param contain_side:
        :param minmax_normalized: set minmax_normalized to False when using Fourier Transform
        :param enable_degradation: Enable random 256Hz->128Hz degradation during training
        :param degradation_prob: Probability of degrading a sample (default 0.3 = 30%)
        """
        # data = (start_datetime, eeg, emg, mot, [lbs], mxs)
        # self.is_infer = is_infer
        self.inp_path = dump_path
        self.w_out = w_out
        self.contain_side = contain_side
        self.minmax_normalized = minmax_normalized
        self.enable_degradation = enable_degradation
        self.degradation_prob = degradation_prob

        # Initialize degradation transform if enabled
        if self.enable_degradation and not is_infer:
            self.degrader = SamplingRateDegradation(orig_freq=256, new_freq=128)
        else:
            self.degrader = None
        data = joblib.load(dump_path)
        if len(data) == 6:
            self.start_datetime, self.eeg, self.emg, self.mot, self.lbs, self.mxs = data
            self.is_infer = False
        elif len(data) == 5:
            self.start_datetime, self.eeg, self.emg, self.mot, self.mxs = data
            self.lbs = []
            self.is_infer = True
        else:
            raise ValueError('Error in EEGDataset. Dump file length should be 6 or 5')
        # if not is_infer:
        #     self.start_datetime, self.eeg, self.emg, self.mot, self.lbs, self.mxs = joblib.load(dump_path)
        # else:
        #     self.start_datetime, self.eeg, self.emg, self.mot, self.mxs = joblib.load(dump_path)
        #     self.lbs = []
        #print(len(self.eeg[0]))
        #exit(-1)
        self.segment_length = params.MAX_SEQ_SIZE
        assert self.segment_length == len(self.eeg[0])


    def __len__(self):
        return len(self.start_datetime)

    def __getitem__(self, idx):
        if self.contain_side == 'none':
            seqs = self._getseq_idx(idx)
            lbs = self._getlb_idx(idx)
            lbs_binary = self._getlb_binary_idx(idx)
        else:
            seqs, lbs, lbs_binary = [[], [], []]
            if self.contain_side == 'right':
                for i in range(idx, idx + self.w_out):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
                    lbs_binary.append(self._getlb_binary_idx(i))
            elif self.contain_side == 'left':
                for i in range(idx - self.w_out + 1, idx + 1):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
                    lbs_binary.append(self._getlb_binary_idx(i))
            elif self.contain_side == 'both':
                pos_shift = params.POS_IDX
                for i in range(idx - pos_shift, idx + pos_shift + 1):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
                    lbs_binary.append(self._getlb_binary_idx(i))
            seqs = torch.concat(seqs, dim=-1)
            lbs = torch.stack(lbs)
            lbs_binary = torch.stack(lbs_binary)

        # Apply random degradation during training (not inference)
        if self.degrader is not None and torch.rand(1).item() < self.degradation_prob:
            seqs = self.degrader(seqs)  # [3, 5120] -> [3, 2560]

        return seqs, lbs, lbs_binary

    def _getseq_idx(self, idx):
        if idx < 0 or idx >= self.__len__():
            eeg = torch.zeros(self.segment_length, dtype=torch.float32)
            emg = torch.zeros(self.segment_length, dtype=torch.float32)
            mot = torch.zeros(self.segment_length, dtype=torch.float32)
        else:
            if self.minmax_normalized:
                eeg = torch.tensor(self.eeg[idx], dtype=torch.float32) / self.mxs[0]
                emg = torch.tensor(self.emg[idx], dtype=torch.float32) / self.mxs[1]
                mot = torch.tensor(self.mot[idx], dtype=torch.float32) / self.mxs[2]
            else:
                eeg = torch.tensor(self.eeg[idx], dtype=torch.float32)
                emg = torch.tensor(self.emg[idx], dtype=torch.float32)
                mot = torch.tensor(self.mot[idx], dtype=torch.float32)

        return torch.stack([eeg, emg, mot])

    def _getlb_idx(self, idx):
        lb = torch.zeros(len(LABEL_DICT), dtype=torch.float32)
        if not self.is_infer:
            if idx < 0 or idx >= self.__len__():
                lb_idx = -1
            else:
                lb_idx = self.lbs[idx]
            lb[lb_idx] = 1.0
            return lb
        else:
            # torch.fill(lb, -1)
            for v in lb:
                v = -1
            return lb

    def _getlb_binary_idx(self, idx):
        lb_binary = torch.zeros(2, dtype=torch.float32)
        if not self.is_infer:
            if 0 <= idx < self.__len__():
                lb_idx = self.lbs[idx]
                if lb_idx % 2 == 0 and lb_idx != -1:
                    lb_binary[0] = 1.0
                else:
                    lb_binary[1] = 1.0
        return lb_binary
