import pdb
import numpy as np
import code_for_hw3_part2 as hw3
import math
#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            # ('model_year', hw3.raw),
            ('origin', hw3.raw)]

features = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            # ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)

# theta, theta_0 = hw3.averaged_perceptron(auto_data, auto_labels, params={'T': 10})
# print('auto data and labels shape', auto_data.shape, auto_labels.shape)

# print(f"Theta: {theta.T}, theta_0: {theta_0}")

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

# acc = hw3.xval_learning_alg(hw3.perceptron, auto_data, auto_labels, 10)
# acc2 = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data, auto_labels, 10)
# print(acc, acc2)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)
# for t in [1, 10, 50]:
#     acc = hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, 10, t=t)
#     acc2 = hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10, t=t)
#     print(f"t: {t}, acc, acc2: {(acc, acc2)}")

# theta, theta0 = hw3.averaged_perceptron(review_bow_data, review_labels, {'T': 10})

# res = dict(zip(dictionary.keys(), theta))

# top = sorted(res.items(), key=lambda x: x[1], reverse=True)
# bottom = sorted(res.items(), key=lambda x: x[1])

# top10 = top[:10]
# bottom10 = bottom[:10]

# top10words = [x[0] for x in top10]
# bottom10words = [x[0] for x in bottom10]
# print(top10words)
# print(bottom10words)


#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)


def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    raise Exception("implement me!")

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    m, n = x.shape
    y = np.zeros((m, 1))
    for i in range(m):
        sum_i = np.sum(x[i], axis=None)
        avg_i = sum_i / n
        y[i,  0] = avg_i
    return y


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    m, n = x.shape
    y = np.zeros((n, 1))
    for i in range(n):
        sum_i = np.sum(x[:, i], axis=None)
        avg_i = sum_i / m
        y[i, 0] = avg_i
    return y


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    row_avg = row_average_features(x)
    top_limit = math.floor(x.shape[0] / 2)
    top = row_avg[:top_limit]
    bottom = row_avg[top_limit:]
    top_avg = np.sum(top, axis=None) / top_limit
    bottom_avg = np.sum(bottom, axis=None) / (x.shape[0] - top_limit)
    z = np.array([[top_avg], [bottom_avg]])
    return z

# use this function to evaluate accuracy
# acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data


# HINT: change the [0] and [1] if you want to access different images
for num1, num2 in [(0, 1)]:
    d0 = mnist_data_all[num1]["images"]
    d1 = mnist_data_all[num2]["images"]
    y0 = np.repeat(-1, len(d0)).reshape(1,-1)
    y1 = np.repeat(1, len(d1)).reshape(1,-1)

    # data goes into the feature computation functions
    data = np.vstack((d0, d1))
    print(data.shape)
    # labels can directly go into the perceptron algorithm
    labels = np.vstack((y0.T, y1.T)).T
    
    n = data.shape[0]
    refined_data_row = np.zeros((n, 28))
    refined_data_col = np.zeros((n, 28))
    for i in range(n):
        row = data[i]
        print(row.shape)
        row_avg = row_average_features(row)
        print(row_avg.shape)
        refined_data_row[i] = row_avg.reshape((28, ))
        print(refined_data_row)
        break
        col_avg = col_average_features(row)
        # print(row_avg)
        refined_data_row[i] = row_avg[:, 0] 
        refined_data_col[i] = col_avg[:, 0]
        
    # refined_data_T = refined_data.T
    print(f"Accuracy row {num1} vs {num2}: {hw3.xval_learning_alg(hw3.perceptron, refined_data_row, labels, 10, 50)}")
    print(f"Accuracy column {num1} vs {num2}: {hw3.xval_learning_alg(hw3.perceptron, refined_data_col, labels, 10, 50)}")
