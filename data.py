import os
import numpy as np
from scipy.io import loadmat


def load_binarized_mnist(dir='.', dataset='train'):

    if dataset == 'train':
        data_train = np.loadtxt(os.path.join(dir, 'data/binarized_mnist/binarized_mnist_train.amat'))
        data_valid = np.loadtxt(os.path.join(dir, 'data/binarized_mnist/binarized_mnist_valid.amat'))
        data = np.concatenate((data_train, data_valid))

    elif dataset == 'test':
        data = np.loadtxt(os.path.join(dir, 'data/binarized_mnist/binarized_mnist_test.amat'))

    else:
        raise Exception('dataset must be "train" or "test"')

    return data


def load_frey_faces(dir='.', test_prop=0.2):

    data = np.transpose(loadmat(os.path.join(dir, 'data/frey_faces/frey_rawface.mat'))['ff'])
    data = np.array([n / float(max(n) + 1) for n in data])

    N = data.shape[0]

    np.random.shuffle(data)

    train = data[int(N*test_prop):]
    test = data[: int(N*test_prop)]

    return train, test
