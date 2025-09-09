import numpy as np

def MSE(actual=0, forecast=0):
    return ((actual - forecast) ** 2).mean()

def MAPE(actual, forecast):
    sum_abs_actual = np.sum(np.abs(actual))
    if sum_abs_actual == 0:
        return np.inf 
    return np.sum(np.abs(actual - forecast)) / sum_abs_actual * 100
