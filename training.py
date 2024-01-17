import jax.lax

from architecture import *
from time import time
from config import test_batch, classes
from jax import vmap, jacobian, pmap
from jax import jit, numpy as np
from functools import partial
from jax.experimental.maps import xmap
# import numpy as np
#
#
# def jit(args, static_argnums):  # debugging
#     return args


@partial(jit, static_argnums=(3, 4))
def gradient_and_cost(x, y, p, batch_size, filt):
    inputs = int(numpy.ceil(numpy.log2(len(x[0]))))  # number of input qubits
    dev = qml.device("default.qubit", wires=inputs)
    # ndev = len(jax.devices())
    # x1 = [x[i*ndev:(i+1)*ndev] for i in range(int(batch_size/ndev))]
    # probs = np.array([]).reshape((0, int(2**(numpy.ceil(numpy.log2(classes))))))
    # jacobians = np.array([]).reshape((0, int(2**(numpy.ceil(numpy.log2(classes)))), len(p)))
    # qnode = jit(qml.QNode(circuit, dev, interface='jax'), static_argnums=2)
    qnode = jit(partial(qml.QNode(circuit, dev, interface='jax'), filters=filt))
    probs = vmap(qnode, in_axes=(0, None))(x, p)
    jacobians = vmap(jacobian(qnode, argnums=1), in_axes=(0, None))(x, p)
    # for i in x1:
    #     pro = pmap(qnode, in_axes=(0, None))(i, p)
    #     probs = np.concatenate((probs, pro))
    #     jac = pmap(jacobian(qnode, argnums=1), in_axes=(0, None))(i, p)
    #     jacobians = np.concatenate((jacobians, jac))
    # probs = jax.lax.map(lambda vec: qnode(vec, p, filt), x)
    # print(probs)
    correctprobs = np.array([])
    correctjac = np.array([])
    for i in range(batch_size):
        correctprobs = np.append(correctprobs, probs[i, y[i]])
        correctjac = np.append(correctjac, jacobians[i, y[i], :])
    # jacobians = jax.lax.map(lambda vec: jax.jacobian(qnode, argnums=1)(vec, p, filt), x)
    correctjac = correctjac.reshape(batch_size, len(p))
    gradients = np.divide(correctjac, correctprobs[:, None])
    g = -np.sum(gradients, axis=0)

    return g, -np.sum(np.log2(correctprobs))


def test_accuracy(testimgs, testlbls, params, classes, filt):
    inputs = int(np.log2(len(testimgs[0])))  # number of input qubits
    read = int(numpy.ceil(numpy.log2(classes)))
    test_sets = len(testlbls)
    prob = np.empty((0, 2 ** read))
    dev = qml.device("default.qubit", wires=inputs)
    qnode = jit(qml.QNode(circuit, dev, interface='jax'), static_argnums=2)
    base = int(time())

    for i in range(int(numpy.ceil(test_sets / test_batch))):
        iterprob = vmap(qnode, in_axes=(0, None, None))(testimgs[i * test_batch:(i + 1) * test_batch], params, filt)
        prob = np.concatenate((prob, iterprob))
        print(f'{i*test_batch} test cases evaluated at {int(time()) - base} seconds')

    predictions = np.argmax(prob, axis=1)
    accuracy = 100 * np.sum(predictions == testlbls) / test_sets
    classaccuracy = numpy.empty(classes)
    for i in range(classes):
        classaccuracy[i] = 100 * np.sum((predictions == testlbls) & (testlbls == i)) / np.sum(testlbls == i)

    unique, counts = np.unique(np.column_stack((testlbls, predictions)), return_counts=True, axis=0)
    confusion = numpy.zeros((classes, 2 ** read))

    for i in range(len(counts)):
        confusion[unique[i, 0], unique[i, 1]] = counts[i]

    correctprobs = prob[np.arange(len(prob)), testlbls]

    return accuracy, correctprobs, classaccuracy, confusion


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')