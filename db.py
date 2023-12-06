from datetime import datetime
from utils import convert_time
import torch
import joblib
import params
from sklearn.metrics import roc_auc_score, average_precision_score
t = torch.arange(10)
h = torch.fft.fft(t)
print(h)
print(torch.real(h))