from emnist import extract_training_samples, extract_test_samples
import numpy
import numpy as np
from jax import numpy as np
from config import classes
import mnist_reader

#
# def jit(args):  # debugging
#     return args


def load(samples, train_sets, filters):
    if samples == 'fashion':
        x0, trainlbls = mnist_reader.load_mnist('datasets/fashion', kind='train')
        w0, testlbls = mnist_reader.load_mnist('datasets/fashion', kind='t10k')
        x0 = numpy.reshape(x0, (len(x0), 28, 28))
        w0 = numpy.reshape(w0, (len(w0), 28, 28))
    elif samples == 'kuzushiji':
        x0, trainlbls = mnist_reader.load_mnist('datasets/kmnist', kind='train')
        w0, testlbls = mnist_reader.load_mnist('datasets/kmnist', kind='t10k')
        x0 = numpy.reshape(x0, (len(x0), 28, 28))
        w0 = numpy.reshape(w0, (len(w0), 28, 28))
    else:
        x0, trainlbls = extract_training_samples(samples)
        w0, testlbls = extract_test_samples(samples)


    if len(np.unique(trainlbls)) != classes:
        pruned = np.zeros(len(trainlbls)).astype(int)
        for i in range(classes):
            pruned = np.array((trainlbls == i)).astype(int) | pruned
        indices = np.nonzero(pruned)
        x0 = x0[indices]
        trainlbls = trainlbls[indices]
        pruned = np.zeros(len(testlbls)).astype(int)
        for i in range(classes):
            pruned = np.array((testlbls == i)).astype(int) | pruned
        indices = np.nonzero(pruned)
        w0 = w0[indices]
        testlbls = testlbls[indices]

    train_sets = min(train_sets, len(x0))
    test_sets = len(w0)

    imgshape = numpy.shape(x0[0])  # shape of input data
    padshape = tuple(
        [int(2 ** np.ceil(np.log2(i))) for i in imgshape])  # set padded size to 2^ceil(log(d)) in each dimension

    trainimgs = numpy.zeros(
        (train_sets,) + padshape)  # save full training input shape as a 3-tuple. works if images are not 2d
    trainimgs[:, (padshape[0] - imgshape[0]) // 2:(padshape[0] + imgshape[0]) // 2,
    (padshape[1] - imgshape[1]) // 2:(padshape[1] + imgshape[1]) // 2] = x0[:train_sets]  # pad the inputs
    trainvecs = numpy.tile(trainimgs, filters)  # tile padded images for multiple filters

    testimgs = numpy.zeros((test_sets,) + padshape)
    testimgs[:, (padshape[0] - imgshape[0]) // 2:(padshape[0] + imgshape[0]) // 2,
    (padshape[1] - imgshape[1]) // 2:(padshape[1] + imgshape[1]) // 2] = w0[:train_sets]
    testvecs = numpy.tile(testimgs, filters)  # same for test images

    resize = padshape[0] * padshape[1] * filters

    trainvecs = (np.array(np.reshape(trainvecs[:train_sets], (train_sets, resize)))).astype(
        float)  # unroll each entry in trainimgs
    trainlbls = np.array(trainlbls[:train_sets])  # take correct num of labels

    testimgs = (np.array(np.reshape(testvecs[:test_sets], (test_sets, resize)))).astype(float)
    testlbls = np.array(testlbls[:test_sets])

    return trainvecs, trainlbls, testimgs, testlbls
