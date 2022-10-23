from positive import sol_positive
import numpy as np

def best_separator(data, ths, th0s, labels):
    A = sol_positive(data, ths, np.tile(th0s.T, (1, data.shape[1])))
    B = A == labels
    index = np.argmax(np.sum(B, axis=1))
    return (ths[:,index:index + 1], th0s[:,index:index + 1])


## Solution
def cv(value_list):
    A = np.array([value_list]).T
    return A

def score_mat(data, labels, ths, th0s):
   pos = np.sign(np.dot(np.transpose(ths), data) + np.transpose(th0s))
   return np.sum(pos == labels, axis = 1, keepdims = True)

def best_separator(data, labels, ths, th0s):
   best_index = np.argmax(score_mat(data, labels, ths, th0s))
   return cv(ths[:,best_index]), th0s[:,best_index:best_index+1] # cv is column vector
