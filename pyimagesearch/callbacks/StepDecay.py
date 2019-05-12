import numpy as np
from math import exp


def step_decay(epoch):
    initAlpha = 0.01
    factor = 0.5
    dropEvery = 10
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    return float(alpha)


def exp_decay(epoch):
    t = epoch
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * exp(-k*t)
    return lrate