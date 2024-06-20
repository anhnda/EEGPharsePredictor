import yaml
import numpy as np

ar  = np.array([0.6, 0.3, 0.1])
print(np.argwhere(ar < 0.5).reshape(-1))