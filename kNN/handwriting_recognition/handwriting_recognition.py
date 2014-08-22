import sys;sys.path.append("..")
import kNN
import numpy
import os


def img_to_vector(img_fn, label=0):
    """Read the first 32 characters of the first 32 rows of an image file.

    @return <ndarray>: a 1x(1024+1) numpy array with data and label, while the
                       label is defaults to 0.
    """
    return_array = []

    for line in open(img_fn).readlines()[:32]:
        return_array.append(line[:32])

    return_array.append(label)

    return numpy.array(return_array, dtype="32f4").view('f4').reshape(1, 1025)


if __name__ == "__main__":
    training_set_files = os.listdir(r"./trainingDigits")
    
    # initiate a matrix, don't forget to allocate the space for the label
    training_set = numpy.zeros((len(training_set_files), 1025))

    for i in xrange(len(training_set_files)):
        training_set[i, :] = img_to_vector(r"./trainingDigits/" + training_set_files[i], training_set_files[i].split('_')[0])

    knn = kNN.kNN(10, training_set, False)
    knn.classify()
    print knn.training_set


