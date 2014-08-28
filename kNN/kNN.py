#-*- coding: utf-8 -*
"""
@author: Jia Shi, j5shi@live.com

@version: 0.2.0

@revision history:
    - 2014-08-06 2:02:36 PM, Jia Shi, file created.
    - 2014-08-20 4:38:16 PM, Jia Shi, add normalization option to training set.
    - 2014-08-25 3:14:38 PM, Jia Shi, fixed some small bugs and unified some interfaces.

@reference:

    - Sahibsingh A. Dudani, "The Distance-Weighted k-Nearest-Neighbors Rule",
      IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS, pp. 325-327, APRIL 1976
"""
import numpy


class kNN(object):

    """
    A distance weighted k-Nearest Neighbors Algorithm.
    """

    def __init__(self, k, training_set, normalize=True):
        """
        @k                <int>: the number of nearest neighbors, must be a positive integer.
        @training_set <ndarray>: the training set should be a 2D array of floats, and
                                 the last column must be the label.
        @normalize       <bool>: if true, all training sets will be normalized.
        """
        if not isinstance(training_set, numpy.ndarray):
            raise TypeError("training set should be numpy array type.")
        else:
            self.training_set = training_set

        self.k = k
        self.normalize = normalize

    def __setattr__(self, name, value):
        if name == 'k' and (value < 0 or value > self.training_set.shape[0]):
            raise ValueError("k out of range.")
        if name == 'normalize' and value is True:
            self.training_set = self.normalize_training_set(self.training_set)

        object.__setattr__(self, name, value)

    def normalize_training_set(self, training_set):
        """Normalize the given data set.

        @training_set <ndarray>: the training set to be normalized.
        """
        min_value_set = training_set.min(axis=0)
        min_value_set[-1] = 0               # skip the label normalization
        self.min_value_set = min_value_set

        max_value_set = training_set.max(axis=0)
        max_value_set[-1] = 1               # skip the label normalization
        self.max_value_set = max_value_set

        ranges = max_value_set - min_value_set
        self.ranges = ranges

        augment = lambda vector: numpy.tile(vector, (training_set.shape[0], 1))
        data_set_norm = (training_set - augment(min_value_set)) / augment(ranges)

        return data_set_norm

    def __normalize_vector(self, vector):
        """Normalize a single training vector

        @vector <ndarray>: a single training vector
        """

        if not isinstance(vector, numpy.ndarray):
            vector = numpy.array(vector, dtype='f4')

        return (vector - self.min_value_set) / self.ranges

    def get_euclidian_distance(self, vec0, vec1):
        """Return the Euclidian distance of two vectors, all
        vectors must be in same dimensions.

        @vec0 <ndarray>: a list of integers.
        @vec1 <ndarray>: a list of integers.
        """
        # skip the calculate of distances of labels
        return numpy.linalg.norm(vec0[:-1] - vec1[:-1])

    def get_label(self, vector):
        """Return the label of a vector.

        @vector [<ndarray>, ...]: a 1xN numpy ndarray object.
        """
        return vector[-1]

    def get_k_nearest_neighbors(self, vec_to_classify):
        """Get the k nearest neighbors of the given vector to be
        classified.

        @vec_to_classify           <ndarray>: the vector be classified, must be in the
                                              same dimension with the vector in training set.
        @return  [(<ndarray>, <float>), ...]: return a sorted list of tuples of neighbors and distance.
        """
        nei, dist = range(2)
        neighbors = [(neighbor, self.get_euclidian_distance(vec_to_classify, neighbor)) for neighbor in self.training_set]

        # sorting based on distances
        return sorted(neighbors, key=lambda neighbor: neighbor[dist])[:self.k]

    def classify(self, vec_to_classify):
        """Classify the given vector and return the label of the class
        into which it should be cataloged.

        @vec_to_classify [<float>, ...]: the vector be classified, must be in the
                                         same dimension with the vector in training set.
        @return                 <float>: the determined label of the class into which
                                         the vec_to_classify to be cataloged.
        """
        if self.normalize is True:
            vec_to_classify = self.__normalize_vector(vec_to_classify)

        # if k == 1, then skip the majority vote and return the nearest neighbor
        if self.k == 1:
            return self.get_label(self.get_k_nearest_neighbors(vec_to_classify)[0][0])
        # do the distance weighted majority vote
        else:
            k_neighbors = self.get_k_nearest_neighbors(vec_to_classify)

            # distance-weighted rule
            distance_max = k_neighbors[-1][1]
            distance_min = k_neighbors[0][1]
            distance_delta = distance_max - distance_min

            if distance_delta == 0:
                distance_delta = 1

            # the weight decreases with increasing sample-to-distance distance
            k_neighbors = [(neighbor[0], (distance_max - neighbor[1]) / distance_delta) for neighbor in k_neighbors]

            # voting
            class_count = {}
            for label, weight in [(self.get_label(neighbor[0]), neighbor[1]) for neighbor in k_neighbors]:
                class_count[label] = class_count.get(label, 0) + weight

            # sorting based on weight in descending order
            return sorted(class_count.iteritems(), key=lambda item: item[1], reverse=True)[0][0]

if __name__ == "__main__":
    if 0:
        label = {1: 'a',
                 2: 'b',
                 3: 'c',
                 'a': 1,
                 'b': 2,
                 'c': 3}

        knn = kNN(2, numpy.array([[2, 2, 8, 1], [1, 7, 1, 2], [3, 3, 3, 3]], dtype="f4"))

        test_vec = numpy.array([2, 8, 2, 0], dtype='f4')
        # set whatever number to the label (last column), since its unknown
        print knn.get_k_nearest_neighbors(test_vec)
        print label.get(knn.classify(test_vec))
    if 1:
        const = 20
        label = {1: "largeDoses",
                 2: "smallDoses",
                 3: "didntLike",
                 "largeDoses": 1,
                 "smallDoses": 2,
                 "didntLike": 3}

        training_set = numpy.genfromtxt(
            "training_set.txt",
            dtype=[('f1', 'f4'), ('f2', 'f4'), ('f3', 'f4'), ('label', 'f4')],
            comments='#',
            usecols=(0, 1, 2, 3),
            converters={3: lambda x: label[x.strip()]}).view("f4").reshape(-1, 4)

        test_set = training_set[-const:]

        knn = kNN(5, training_set)
        knn1 = kNN(5, training_set, False)

        # test of the kNN algorithm and the comparison of normalized and unnormalized training set
        for test_vec in test_set:
            print label.get(knn.classify(test_vec), "unknown"), label.get(knn1.classify(test_vec), "unknown"), label.get(test_vec[-1])
