from datetime import datetime
from utils import convert_time
import torch
import joblib
import params
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
ar = np.random.random((3,1000))
v = np.ones(3)[:, np.newaxis]
print(ar/v)