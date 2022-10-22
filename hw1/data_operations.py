from positive import positive
import numpy as np

"""
A should be a 1 by 5 array of boolean values, either True or False, indicating for each point in data and corresponding label in labels whether it is correctly classified by hyperplane th = [1, 1], th0 = -2 . That is, return True when the side of the hyperplane (specified by th, th0) that the point is on agrees with the specified label.
"""
# th = [1, 1] , th0 = -2
def check_labels(data, labels):
    return positive(data, np.array([[1], [1]]), -2) == labels # np.array[True or False]    
