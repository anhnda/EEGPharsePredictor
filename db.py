from datetime import datetime
from utils import convert_time
import torch
import joblib
import params
from sklearn.metrics import roc_auc_score, average_precision_score
ar_pred = [[9.1217e-03, 6.4306e-04, 9.8259e-01, 6.2543e-03, 1.2781e-03, 1.1569e-04],
        [7.2408e-02, 1.3401e-01, 1.3735e-02, 1.2923e-02, 1.5813e-01, 6.0879e-01]]
ar_true =[[0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1.]]

print(roc_auc_score(ar_true, ar_pred), average_precision_score(ar_true, ar_pred))

