import numpy as np
import math

def length(col_v):
    return math.sqrt(np.sum(np.dot(col_v.T, col_v)))
