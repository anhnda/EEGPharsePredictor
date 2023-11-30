import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import params
import joblib
import math


class EGGDataset(Dataset):
    def __init__(self, dump_path=params.DUMP_FILE):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        value_seqs, label_seqs, mx, lb_dict = joblib.load(dump_path)
        self.value_seqs = value_seqs
        self.mx = mx
        self.label_seqs = label_seqs
        self.lb_dict = lb_dict
        self.num_class = len(lb_dict)
        self.cls = torch.zeros((params.D_MODEL,1))

    def __len__(self):
        return len(self.value_seqs)

    def get_num_class(self):
        return self.num_class

    def __getitem__(self, idx):
        value_seq = self.value_seqs[idx]
        assert len(value_seq) == params.MAX_SEQ_SIZE
        label_id = self.label_seqs[idx]
        label_ar = torch.zeros(self.num_class)
        label_ar[label_id] = 1
        value_seq = torch.tile(torch.from_numpy(np.asarray(value_seq)) / self.mx, (params.D_MODEL, 1))
        # print("VShape: ",value_seq.shape)
        value_seq = torch.hstack([self.cls, value_seq]).transpose(0,1)
        return value_seq, label_ar
