import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps: float = 1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# Backward compatible aliases (so old code doesn't break)
_sigmoid = sigmoid
_logit = logit
