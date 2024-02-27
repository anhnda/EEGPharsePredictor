from datetime import datetime
from utils import convert_time
import torch
import joblib
import params
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from my_logger.logger2 import MyLogger
line = "('Best values: ', [tensor(0.6711), 0.5498730001925964, 0.5498730001925964, 8, 8, 0.82654733612179, 0.41197165998790375])"
print(line.split(","))
2,6,7