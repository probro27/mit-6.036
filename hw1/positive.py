from signed_distance import signed_dist
import numpy as np
"""
Write a Python function that takes as input

a column vector x
a column vector th that is of the same dimension as x
a scalar th0
and returns

+1 if x is on the positive side of the hyperplane encoded by (th, th0)
0 if on the hyperplane
-1 otherwise.
The answer should be a 2D array (a 1 by 1). Look at the \href{https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sign.html}{sign}sign function. Note that you are allowed to use any functions defined in week 1's exercises.
"""
def positive(x, th, th0):
    return np.sign(signed_dist(x, th, th0))

def sol_positive(x, th, th0):
    return np.sign(np.dot(th.T, x) + th0)
