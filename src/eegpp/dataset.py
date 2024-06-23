import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import joblib

from .import params


class EGGDataset(Dataset):
    def __init__(self, dump_path=None, tile_seq=False, cls_pad=True, side_flag=False, with_label=True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.with_label = with_label
        self.dump_path = dump_path
        if self.with_label:
            value_seqs, label_seqs, mx, misc = joblib.load(dump_path)
        else:
            value_seqs, mx, misc = joblib.load(dump_path)
            label_seqs = []
        lb_dict = misc["LB_DICT"]
        self.misc = misc
        # print(len(value_seqs), len(value_seqs[0]), len(value_seqs[0][7786]), len(value_seqs[1][7786]), len(value_seqs[2][7786]), len(label_seqs))
        self.value_seqs = value_seqs
        self.mx = np.asarray(mx)[:, np.newaxis]
        if params.TWO_CHAINS:
            self.mx = self.mx[:2, :]
        # print(self.mx)
        self.label_seqs = label_seqs
        self.lb_dict = lb_dict
        self.idx_2lb = {v: k for k, v in lb_dict.items()}
        self.idx_2lb.update({6: "X"})
        self.num_class = params.NUM_CLASSES
        self.cls = torch.zeros((params.D_MODEL, 1))
        self.tile_seq = tile_seq
        self.cls_pad = cls_pad
        self.side_flag = side_flag

    def __len__(self):
        return len(self.value_seqs[0])

    def get_num_class(self):
        return self.num_class

    def __getlb_idx(self, idx):
        if not self.with_label:
            return [-1,-1]
        if idx < 0 or idx >= self.__len__():
            return [-1, -1]
        return self.label_seqs[idx]

    def __getseq_idx(self, idx):
        # print(idx)
        if idx < 0 or idx >= self.__len__():
            if params.THREE_CHAINS:
                value_seq = np.zeros((3, params.MAX_SEQ_SIZE))
            elif params.TWO_CHAINS:
                value_seq = np.zeros((2, params.MAX_SEQ_SIZE))
            else:
                value_seq = np.zeros(params.MAX_SEQ_SIZE)
        else:
            if params.THREE_CHAINS:
                value_seq = np.asarray(
                    [self.value_seqs[0][idx], self.value_seqs[1][idx], self.value_seqs[2][idx]]) / self.mx
            elif params.TWO_CHAINS:
                value_seq = np.asarray(
                    [self.value_seqs[0][idx], self.value_seqs[1][idx]]) / self.mx
            else:
                value_seq = np.asarray(self.value_seqs[0][idx]) / self.mx

        value_seq = torch.from_numpy(value_seq)
        if params.OFF_EGG:
            value_seq[0, :] = 0
        if params.OFF_EMG:
            value_seq[1, :] = 0
        if params.OFF_MOT and not params.TWO_CHAINS:
            value_seq[2, :] = 0
        # print("VDS: ", params.OFF_MOT, value_seq[2,-100:])
        return value_seq

    def __getitem__(self, idx):
        value_seq = self.__getseq_idx(idx)
        if params.THREE_CHAINS or params.TWO_CHAINS:
            assert len(value_seq[0]) == params.MAX_SEQ_SIZE
            # value_seq[2].fill(0)
        else:
            assert len(value_seq) == params.MAX_SEQ_SIZE
        if self.with_label:
            label_id, epoch_id = self.label_seqs[idx]
        else:
            epoch_id = idx
            label_id = 0
        label_ar = torch.zeros(self.num_class)
        label_ar[label_id] = 1
        label_windows = [label_id]
        if self.tile_seq:
            value_seq = torch.tile(torch.from_numpy(np.asarray(value_seq)) / self.mx, (params.D_MODEL, 1))
            if self.cls_pad:
                value_seq = torch.hstack([self.cls, value_seq]).transpose(0, 1)
        else:
            if self.side_flag == params.TWO_SIDE:
                value_seqs = []
                label_windows = []
                # Left
                for ii in range(params.POS_ID, 0, -1):
                    value_seqs.append(self.__getseq_idx(idx-ii))
                    label_windows.append(self.__getlb_idx(idx-ii)[0])
                # Main mid
                value_seqs.append(value_seq)
                label_windows.append(label_id)
                # Right
                for ii in range(1, params.POS_ID+1):
                    value_seqs.append(self.__getseq_idx(idx+ii))
                    label_windows.append(self.__getlb_idx(idx + ii)[0])

                value_seq = torch.concat(value_seqs, dim=-1)



            elif self.side_flag == params.LEFT:
                raise 'Deprecated!'
                # value_seq_left = self.__getseq_idx(idx - 1)
                # value_seq = torch.concat((value_seq_left, value_seq), dim=-1)
                # label_windows = [label_id, self.__getlb_idx(idx + 1)[0]]


        label_windows_array = np.zeros((self.num_class, len(label_windows)))
        for i, v in enumerate(label_windows):
            label_windows_array[v, i] = 1

        return value_seq, label_ar, torch.asarray(label_windows), torch.from_numpy(
            label_windows_array).float(), epoch_id
