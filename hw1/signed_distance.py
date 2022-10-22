import numpy as np
from length import length

"""
Write a Python function using numpy operations (no loops!) that takes column vectors (dd by 1) x and th (of the same dimension) and scalar th0 and returns the signed perpendicular distance (as a 1 by 1 array) from the hyperplane encoded by (th, th0) to x. Note that you are allowed to use the "length" function defined in previous coding questions (includig week 1 exercises).
"""
def signed_dist(x, th, th0):
    return (np.dot(th.T, x) / length(th) + th0 / length(th))
