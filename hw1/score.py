from positive import positive
import numpy as np
"""
Write a procedure that takes as input
data: a d by n array of floats (representing n data points in d dimensions)
labels: a 1 by n array of elements in (+1, -1), representing target labels
th: a d by 1 array of floats that together with
th0: a single scalar or 1 by 1 array, represents a hyperplane
and returns the number of points for which the label is equal to the output of the positive function on the point.
"""
def score(data, labels, th, th0):
    A = positive(data, np.array(th), th0) == labels
    return np.sum(A)
