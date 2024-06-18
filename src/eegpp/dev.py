import torch
import os

C_DIR = os.path.dirname(os.path.abspath(__file__))

device1 = None
if torch.backends.mps.is_available():
    device0 = torch.device("mps")
elif torch.cuda.is_available():
    device0 = torch.device("cuda:0")
else:
    device0 = torch.device("cpu")


def get_device(dv=None):
    global device1
    if dv is not None:
        device1 = torch.device(dv)

    if device1 is not None:
        return device1
    else:
        return device0
