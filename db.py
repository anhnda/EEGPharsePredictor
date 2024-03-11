from datetime import datetime
from utils import convert_time
import torch
import joblib
import params
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from my_logger.logger2 import MyLogger
fout = open("long_out.txt", "w")
for i in range(10000):
    fout.write("%s\n" % (i+1))
fout.close()