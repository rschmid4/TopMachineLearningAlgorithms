"""
@author: Jia Shi, j5shi@live.com

@version: 0.2

@revision history:
    - 2014-08-06 2:02:36 PM, file created.
    - 2014-08-20 4:38:16 PM, add normalization option to training set.

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
        @k                <int>: the number of nearest neighbors, must be positive.
        @training_set <ndarray>: the training set should be a 2D array, and the last column must
                                 be the label.
        @label        <ndarray>: the label of the training set, will be convert to numpy.ndarray.
        @normalize      boolean: if true, all training sets will be normalized.
        """
        # in case the training_set is not ndarray type, convert it to ndarray
        if isinstance(training_set, numpy.ndarray):
            self.training_set = training_set
        else:
            self.training_set = numpy.array(training_set, dtype='f4')

        # in case the training set is structured array, convert to normal array
        if len(self.training_set.shape) == 1:
            self.training_set = self.training_set.view('f4').reshape(self.training_set.shape[0], -1)
        
        self.k = k
        self.normalize = normalize

    def __setattr__(self, name, value):
        if name == 'k':
            if value < 0 or value > self.training_set.shape[0]:
                raise ValueError("k should be an integer and in range [1, %s]" % self.training_set.shape[0])
        if name == 'normalize' and value is True:
            self.training_set = self.normalize_training_set(self.training_set)
        object.__setattr__(self, name, value)
    

    def normalize_training_set(self, data_set):
        """ normalize the given data set.
        
        @data_set <ndarray>: the data set to be normalized.
        """
        min_value_set = data_set.min(axis=0)
        min_value_set[-1] = 0               # skip the label normalization
        self.min_value_set = min_value_set

        max_value_set = data_set.max(axis=0)
        max_value_set[-1] = 1               # skip the label normalization
        self.max_value_set = max_value_set

        ranges = max_value_set - min_value_set
        self.ranges = ranges

        augment = lambda vector: numpy.tile(vector, (data_set.shape[0], 1))
        data_set_norm = (data_set - augment(min_value_set)) / augment(ranges)
       
        return data_set_norm

    def __normalize_vector(self, vector):
        """normalize a single training vector
        
        @vector <ndarray>: a single training vector
        """

        if not isinstance(vector, numpy.ndarray):
            vector = numpy.array(vector, dtype='f4')

        return (vector - self.min_value_set) / self.ranges

    def get_euclidian_distance(self, vec0, vec1):
        """ return the Euclidian distance of two vectors, all
        vectors must be in same dimensions.

        @vec0 <ndarray>: a list of integers.
        @vec1 <ndarray>: a list of integers.
        """
        # skip the calculate of distances of labels
        return sum([(a - b) ** 2 for a, b in zip(vec0[:-1], vec1[:-1])]) ** 0.5

    def get_k_nearest_neighbors(self, vec_to_classify):
        """ get the k nearest neighbors of the given vector to be
        classified. k can be redefined in each call.

        @vec_to_classify          <ndarray>: the vector be classified, must be in the
                                             same dimension with the vector in training set.
        @return  [(neighbor, distance),...]: return a sorted list of tuples.
        """
        nei, dist = range(2)
        neighbors = [(neighbor, self.get_euclidian_distance(vec_to_classify, neighbor)) for neighbor in self.training_set]

        # sorting based on distances
        return sorted(neighbors, key=lambda neighbor: neighbor[dist])[:self.k]

    def classify(self, vec_to_classify):
        """ classify the given vector and return the label of the class
        into which it should be cataloged.

        @vec_to_classify [float,...]: the vector be classified, must be in the
                                      same dimension with the vector in training set.
        @return              "label": the determined label of the class into which
                                      the vec_to_classify to be cataloged.
        """
        if self.normalize is True:
            vec_to_classify = self.__normalize_vector(vec_to_classify)
        
        # if k == 1, then skip the majority vote and return the nearest neighbor
        if self.k == 1:
            return self.get_k_nearest_neighbors(vec_to_classify)[0][0][-1]
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
            for label, weight in [(neighbor[0][-1], neighbor[1]) for neighbor in k_neighbors]:
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
                 'c': 3 }

        knn = kNN(2, [[2, 2, 8, 1], [1, 7, 1, 2], [3, 3, 3, 3]])

        test_vec = numpy.array([6, 1, 4, 0], dtype='f4')
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
                converters={3: lambda x: label[x.strip()]})

        test_set = training_set[-const:].view('f4').reshape(const, -1)

        knn = kNN(5, training_set)
        knn1 = kNN(5, training_set, False)

        # test of the kNN algorithm and the comparison of normalized and unnormalized training set
        for test_vec in test_set:
            print label.get(knn.classify(test_vec), "unknown"), label.get(knn1.classify(test_vec), "unknown"), label.get(test_vec[-1])
