"""
Experiment summary
------------------
I would like to see how KNN classifies death rates
for cities in each state. I would like to see if it could
tell what cities are in what states based on death rate.

I also would like to see if these results and the results
I get after

"""

import sys

sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
from scipy.stats import multivariate_normal


########################################## From my implementation of Homework 2 ##############################

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    # sqr((x1-y1)^2 + (x2-y2)^2)

    D = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i][j] = np.sqrt(np.dot(X[i], X[i]) - 2 * np.dot(X[i], Y[j]) + np.dot(Y[j], Y[j]))

    return D


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    # abs(x1-y1) + abs(x2-y2)
    D = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i][j] = np.linalg.norm(X[i] - Y[j], ord=1)

    return D


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    # x(transposed)dot y /(sqrt of all x^2)(sqrt of all y^2)

    D = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            D[i][j] = 1 - (np.dot(np.transpose(X[i]), Y[j]) / (np.linalg.norm(X[i] * np.linalg.norm(Y[j]))))
    return D


class KNearestNeighbor():
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances,
        if  'cosine', use cosine_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3
        closest neighbors are:
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]
            ]
        And the aggregator is 'mean', applied along each dimension, this will return for
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels.
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.real_neighbors = None
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = None
        self.targets = None

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional!

        HINT: One use case of KNN is for imputation, where the features and the targets
        are the same. See tests/test_collaborative_filtering for an example of this.

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples,
                n_dimensions).
        """
        self.features = features
        self.targets = targets


    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor.
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """

        labels = np.zeros((features.shape[0], 1))
        k = self.n_neighbors

        args = np.zeros((k, 1))
        if self.distance_measure == 'euclidean':
            distance = euclidean_distances(features, self.features)
        elif self.distance_measure == 'manhattan':
            distance = manhattan_distances(features, self.features)
        else:
            distance = cosine_distances(features, self.features)
        n = 0
        self.real_neighbors = np.zeros([features.shape[0], k])
        for d in distance:
            test_arg = np.argsort(d, axis=0)
            self.real_neighbors[n] = test_arg[:k]
            for i in range(k):
                if ignore_first:
                    args[i] = self.targets[test_arg[i + 1]]
                else:
                    args[i] = self.targets[test_arg[i]]

            if self.aggregator == 'mean':
                label_int = np.mean(args, axis=0)
            elif self.aggregator == 'median':
                label_int = np.median(args, axis=0)
            else:
                label_int = np.zeros((1, args.shape[1]))
                for i in range(args.shape[1]):
                    mode_stuff = np.unique(args[:, i], return_counts=True)
                    nums = mode_stuff[0]
                    count = np.amax(mode_stuff[1])
                    m = 0
                    for j in range(mode_stuff[1].shape[0]):
                        if count == mode_stuff[1][m]:
                            break
                        m += 1
                    label_int[0][i] = nums[m]
                label_int = label_int[0][0]

            labels[n] = label_int
            n += 1
        return labels


##############################################################################################################
# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 5
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US.csv')

confirmed = data.load_csv_data(confirmed)

state_dict = {}
features_confirmed = []
targets_confirmed = []
i = 0;

for val in np.unique(confirmed["Province_State"]):
    state_dict.update({i: val})
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)

    cases, labels = data.get_cases_chronologically(df)
    label = i

    new_labels = np.ones(labels.shape[0])*i

    features_confirmed.append(cases)
    targets_confirmed.append(new_labels)
    i += 1

features_confirmed = np.concatenate(features_confirmed, axis=0)

targets_confirmed = np.concatenate(targets_confirmed, axis=0)
unique = np.unique(targets_confirmed, return_counts=True)
small_values = np.where(unique[1] <=5)

numbers = np.arange(features_confirmed.shape[0])
np.random.shuffle(numbers)
new_features_confirmed = np.copy(features_confirmed)
new_targets_confirmed = np.copy(targets_confirmed)
new_desc = []

for i in numbers:
    new_features_confirmed[i] = features_confirmed[i]
    new_targets_confirmed[i] = targets_confirmed[i]

other_data = new_features_confirmed[:, :8]
print(new_features_confirmed.shape)
for j in range(8):
    new_features_confirmed = np.delete(new_features_confirmed, 0, 1)

train_features_confirmed = new_features_confirmed[:2495]
train_targets_confirmed = new_targets_confirmed[:2495]

test_features_confirmed = new_features_confirmed[2495:]
test_targets_confirmed = new_targets_confirmed[2495:]
test_other = other_data[2495:]

print(train_features_confirmed.shape)

knearest_learner = KNearestNeighbor(7)
knearest_learner.fit(train_features_confirmed, train_targets_confirmed)

prediction_confirmed = knearest_learner.predict(test_features_confirmed)
predictions = {}
print(train_targets_confirmed)
print(state_dict[test_targets_confirmed[0]])
print(test_targets_confirmed[0])
print(test_features_confirmed[0])
print(test_other[0])


for i in range(prediction_confirmed.shape[0]):
    state = state_dict[int(prediction_confirmed[i][0])]
    real_state = state_dict[int(test_targets_confirmed[i])]

    predict = test_other[i][1]

    predictions.update({str(predict) + ", " + str(real_state): state})


with open('results/knn_confirmed.json', 'w') as f:
    json.dump(predictions, f, indent=4)