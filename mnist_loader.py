try:
    import cPickle as pickle
except ImportError:
    import pickle
import struct
import numpy as np
import sys

def vectorized_result(j):
    """
    Vectorize the result given a number j.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def to_ascii(nd):
    """
    Converts a single point of data (x, y) where x is the flattened data point
    and y is the vectorized result of what number the data point is, into an
    ascii art image. This makes it easier to debug and see what is going on
    with the dataset
    """
    print(np.argmax(nd[1]))
    for row in range(28):
        for col in range(28):
            if nd[0][row*28 + col][0] >= 0 and nd[0][row*28 + col][0] <= 0.35:
                sys.stdout.write("  ")
            else:
                sys.stdout.write("@@")

        sys.stdout.write("\n")

def load_set():
    """
    Loads the pickle file into training, validation, and testing sets.
    """
    with open('mnist_data', 'rb') as f:
        training, validation, testing = pickle.load(f)
        f.close()
        return training, validation, testing

def parse_image():
    """
    Converts the mnist dataset into a pickle.
    Run only once.
    """
    LABELS = ['t10k-images.idx3-ubyte',
              't10k-labels.idx1-ubyte',
              'train-images.idx3-ubyte',
              'train-labels.idx1-ubyte']

    TRAIN_IMAGE = LABELS[2]
    TRAIN_LABELS = LABELS[3]
    TEST_IMAGE = LABELS[0]
    TEST_LABELS = LABELS[1]

    f_image = open('mnist/' + TRAIN_IMAGE, 'rb')
    f_label = open('mnist/' + TRAIN_LABELS, 'rb')

    i_magic, n_images, n_row, n_col = struct.unpack(">4I", f_image.read(16))
    l_magic, n_labels = struct.unpack(">2I", f_label.read(8))

    assert n_labels == n_images

    training = [tuple([np.array([[i/255.0]
                                 for _ in range(n_row)
                                 for i in struct.unpack(">%dB" % (n_col), f_image.read(n_col))]),
                       vectorized_result(struct.unpack(">B", f_label.read(1))[0])])
                for _ in range(n_images-10000)]

    validation = [tuple([np.array([[i/255.0]
                                   for _ in range(n_row)
                                   for i in struct.unpack(">%dB" % (n_col), f_image.read(n_col))]),
                         vectorized_result(struct.unpack(">B", f_label.read(1))[0])])
                  for _ in range(n_images-10000, n_images)]

    f_image.close()
    f_label.close()

    f_image = open('mnist/' + TEST_IMAGE, 'rb')
    f_label = open('mnist/' + TEST_LABELS, 'rb')

    i_magic, n_images, n_row, n_col = struct.unpack(">4I", f_image.read(16))
    l_magic, n_labels = struct.unpack(">2I", f_label.read(8))

    testing = [tuple([np.array([i/255.0
                                  for _ in range(n_row)
                                  for i in struct.unpack(">%dB" % (n_col), f_image.read(n_col))]),
                      vectorized_result(struct.unpack(">B", f_label.read(1))[0])])
               for _ in range(n_images)]

    with open('mnist_data', 'wb') as f:
        pickle.dump((training, validation, testing), f)
        f.close()

    return training, validation, testing


if __name__ == '__main__':
    parse_image()
