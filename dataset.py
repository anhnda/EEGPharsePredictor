import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import params
import joblib
import math


class EGGDataset(Dataset):
    def __init__(self, dump_path=params.DUMP_FILE, tile_seq=False, cls_pad=True, two_side=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        value_seqs, label_seqs, mx, lb_dict = joblib.load(dump_path)
        # print(len(value_seqs), len(value_seqs[0]), len(value_seqs[0][7786]), len(value_seqs[1][7786]), len(value_seqs[2][7786]), len(label_seqs))
        self.value_seqs = value_seqs
        self.mx = mx
        self.label_seqs = label_seqs
        self.lb_dict = lb_dict
        self.idx_2lb = {v: k for k, v in lb_dict.items()}
        self.num_class = len(lb_dict)
        self.cls = torch.zeros((params.D_MODEL, 1))
        self.tile_seq = tile_seq
        self.cls_pad = cls_pad
        self.two_side = two_side

    def __len__(self):
        return len(self.label_seqs)

    def get_num_class(self):
        return self.num_class

    def __getseq_idx(self, idx):
        # print(idx)
        if idx < 0 or idx >= self.__len__():
            value_seq = np.zeros(params.MAX_SEQ_SIZE)
        else:
            if params.THREE_SIGNAL_TYPES:
                value_seq = np.asarray(
                    [self.value_seqs[0][idx], self.value_seqs[1][idx], self.value_seqs[2][idx]]) / self.mx
            else:
                value_seq = np.asarray(self.value_seqs[0][idx]) / self.mx

        value_seq = torch.from_numpy(value_seq)
        return value_seq

    def __getitem__(self, idx):
        value_seq = self.__getseq_idx(idx)
        if params.THREE_SIGNAL_TYPES:
            assert len(value_seq[0]) == params.MAX_SEQ_SIZE
        else:
            assert len(value_seq) == params.MAX_SEQ_SIZE
        label_id = self.label_seqs[idx]
        label_ar = torch.zeros(self.num_class)
        label_ar[label_id] = 1
        if self.tile_seq:
            value_seq = torch.tile(torch.from_numpy(np.asarray(value_seq)) / self.mx, (params.D_MODEL, 1))
            if self.cls_pad:
                value_seq = torch.hstack([self.cls, value_seq]).transpose(0, 1)
        else:
            if self.two_side:
                value_seq_left = self.__getseq_idx(idx - 1)
                value_seq_right = self.__getseq_idx(idx + 1)
                value_seq = torch.concat((value_seq_left, value_seq, value_seq_right))
            pass

        return value_seq, label_ar
