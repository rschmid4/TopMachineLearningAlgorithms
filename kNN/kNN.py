"""
@author: Jia Shi, j5shi@live.com

@version: 0.1

@revision history:
    - 2014-08-06 2:02:36 PM, file created.

@reference:

    - Sahibsingh A. Dudani, "The Distance-Weighted k-Nearest-Neighbors Rule",
      IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS, pp. 325-327, APRIL 1976
"""


class kNN(object):

    """
    A distance weighted k-Nearest Neighbors Algorithm.
    """

    def __init__(self, k, training_set, labels):
        """
        @k                        <int>: the number of nearest neighbors, must be positive.
        @training_set [[int, ...], ...]: the training set.
        @label           ["label", ...]: the label of the training set.
        """
        self.training_set = training_set
        self.k = k
        self.labels = labels
        self.label_mapping = {key: val for (key, val) in zip(range(len(training_set)), labels)}

    def __setattr__(self, name, value):
        if name == 'k':
            if value < 0 or value > len(self.training_set):
                raise ValueError("k should be a positive integer and no bigger than number of vectors in training set")
        object.__setattr__(self, name, value)

    def get_label(self, vec):
        """ get the label of a vector in training set.

        @vec <list>: a vector in training set.
        """
        return self.label_mapping.get(self.training_set.index(vec), None)

    def get_distance(self, vec0, vec1):
        """ return the Euclidian distance of two vectors, all
        vectors must be in same dimensions.

        @vec0 <list>: a list of integers.
        @vec1 <list>: a list of integers.
        """
        return sum([(a - b) ** 2 for a, b in zip(vec0, vec1)]) ** 0.5

    def get_k_nearest_neighbors(self, vec_to_classify):
        """ get the k nearest neighbors of the given vector to be
        classified. k can be redefined in each call.

        @vec_to_classify          [int,...]: the vector be classified, must be in the
                                             same dimension with the vector in training set.
        @return  [(neighbor, distance),...]: return a sorted list of tuples.
        """
        neighbors = [(neighbor, self.get_distance(vec_to_classify, neighbor)) for neighbor in self.training_set]

        # sorting based on distances
        return sorted(neighbors, key=lambda neighbor: neighbor[1])[:self.k]

    def classify(self, vec_to_classify):
        """ classify the given vector and return the label of the class
        into which it should be cataloged.

        @vec_to_classify [int,...]: the vector be classified, must be in the
                                    same dimension with the vector in training set.
        @return            "label": the determined label of the class into which
                                    the vec_to_classify to be cataloged.
        """
        # if k == 1, then skip the majority vote and return the nearest neighbor
        if self.k == 1:
            return self.get_label(self.get_k_nearest_neighbors(vec_to_classify)[0][0])
        # do the distance weighted majority vote
        else:
            k_neighbors = self.get_k_nearest_neighbors(vec_to_classify)

            # weighting
            distance_max = k_neighbors[-1][1]
            distance_delta = distance_max - k_neighbors[0][1]
            if distance_delta == 0:
                raise Exception("unable to break the tie of same distances!")
            else:
                k_neighbors = [(neighbor[0], (distance_max - neighbor[1]) / distance_delta) for neighbor in k_neighbors]

            # voting
            class_count = {}
            for label, weight in [(self.get_label(neighbor[0]), neighbor[1]) for neighbor in k_neighbors]:
                class_count[label] = class_count.get(label, 0) + weight

            print k_neighbors
            print class_count
            # sorting based on weight in descending order
            return sorted(class_count.iteritems(), key=lambda item: item[1], reverse=True)[0][0]

if __name__ == "__main__":
    knn = kNN(3, [[2, 2, 8], [1, 7, 1], [3, 3, 3]], ['a', 'b', 'c'])
    print knn.classify([0, 0, 0])
