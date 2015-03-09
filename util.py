"""
Utilities for the Deep-Belief-Net Theano implementation.
"""

import logging
import math
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import cPickle as pickle
import theano
from time import time
from datetime import date

log = logging.getLogger(__name__)


def create_minibatches(x, y, size, shuffle=True):
    """
    Default implementation for batching the
    data, override for finer control.

    Returns batched data in the form of a
    list of (x, y) batches if y is not None.
    Otherwise, if y is None, it returns a list
    of x batches.

    :type x: list
    :param x: A list of sentences. Each sentence
    is a list of indices of vocabulary terms.

    :type y: list
    :param y: A list of sentence labels, when using
    the RAE in a supervised fashion, or None when purely
    unsupervised.

    :type size: int, float
    :param size: Desired size of the minibatches. If
    int, then taken as the desired size. If float (0, 1), then
    taken as the desired perecentage x.

    :type shuffle: boolean
    :param shuffle: If or not the trainset should be
    shuffled prior to splitting into batches. If the trainset
    is ordered with regards to classes, shuffling will ensure
    that classes are approximately uniformly represented in
    each minibatch.

    """
    #   convert float size to int size
    if isinstance(size, float):
        size = int(math.ceil(len(x) * size))

    #   if size out of range, ensure appropriate
    size = min(size, len(x))
    size = max(1, size)
    log.info('Creating minibatches, size: %d', size)

    #   shuffle trainset
    if shuffle:
        if y is not None:
            assert len(x) == len(y)
            p = np.random.permutation(len(x))
            x = x[p]
            y = y[p]
        else:
            np.random.shuffle(x)

    #   split x and y into a batch of tuples
    batches_x = []
    batches_y = []
    while True:
        low_ind = len(batches_x) * size
        high_ind = min(low_ind + size, len(x))
        batches_x.append(x[low_ind:high_ind])
        if y is not None:
            batches_y.append(y[low_ind:high_ind])

        if high_ind >= len(x):
            break

    log.info('Created %d minibatches', len(batches_x))

    if y is not None:
        return batches_x, batches_y
    else:
        return batches_x


def try_pickle_load(file_name, zip=None):
    """
    Tries to load pickled data from a file with
    the given name. If unsuccesful, returns None.
    Can compress using Zip.

    :param file_name: File path/name.
    :param zip: If or not the file should be zipped.
        If None, determined from file name.
    """
    if zip is None:
        zip = file_name.lower().endswith("zip")

    try:
        if zip:
            file = ZipFile(file_name, 'r')
            entry = file.namelist()[0]
            data = pickle.load(BytesIO(file.read(entry)))
        else:
            file = open(file_name, "rb")
            data = pickle.load(file)
        log.info('Succesfully loaded pickle %s', file_name)
        return data
    except IOError:
        log.info('Failed to load pickle %s', file_name)
        return None
    finally:
        if 'file' in locals():
            file.close()


def try_pickle_dump(data, file_name, zip=None, entry_name="Data.pkl"):
    """
    Pickles given data tp the given file name.
    Returns True if succesful, False otherwise.

    :param data: The object to pickle.
    :param file_name: Name of file to pickle to.
    :param zip: If or not the file should be zipped.
        If None, determined from file name.
    :param entry_name: If zipping, the name to be used
        for the ZIP entry.
    """
    if zip is None:
        zip = file_name.lower().endswith("zip")

    try:
        log.info('Attempting to pickle data to %s', file_name)
        if zip:
            file = ZipFile(file_name, 'w', ZIP_DEFLATED)
            file.writestr(entry_name, pickle.dumps(data))
        else:
            pickle.dump(data, open(file_name, "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except IOError:
        log.info('Failed to pickle data to %s', file_name)
        return False
    finally:
        if 'file' in locals():
            file.close()


def labels_to_indices(labels):
    """
    Converts an iterable of labels into
    a numpy vector of label indices (zero-based).

    Returns a tuple (indices, vocabulary) so that
    vocabulary[index]=label
    """
    vocab = sorted(set(labels))
    indices = np.array([vocab.index(lab) for lab in labels], dtype=np.int)

    return indices, vocab


def one_hot(indices, count=None):
    """
    Takes a vector of 0 based indices (numpy array) and
    converts it into a matrix of one-hot-encoded
    indices (each index becomes one row).

    For example, if 'indices' is [2, 3], the
    results is:
    [
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ]

    :param indices: The indices to convert.

    :param count: The number elements each one-hot-encoded
        vector should have. If 'None', it is assumed to be
        (indices.max() + 1)
    """

    #   ensure indices is a vector
    indices = indices.reshape(indices.size)

    #   get the max size
    if count is None:
        count = indices.max() + 1
    else:
        assert indices.max() < count

    encoded = np.zeros((indices.size, count), dtype=np.uint8)
    encoded[range(indices.size), indices] = 1

    return encoded


def cost_minimization(inputs, cost, params, epochs, eps, x_mnb, y_mnb):
    """
    Generic cost minimization function (gradient descent) for a
    situaition where given input there are desired outputs.

    :type inputs: iterable of Theano symbolic vars, 2 elements.
    :param inputs: Symblic variables that are inputs to the cost function.
        The iterable needs to consist of two elements, the first is a sym
        variable for minibatch input (x), and the second is a sym for
        minibatch outputs (y).

    :type cost: Theano symbolic variable.
    :param cost: The cost function which needs to be minimized.

    :type params: iterable of theano symbolic vars
    :param params: All the parameters which need to be optimized with
        gradient descent.

    :type epochs: int
    :param epochs: Number of epochs (int) of training.

    :type eps: float
    :param eps: Learning rate.

    :param x_mnb: Trainset split into minibatches. Thus,
        x_mnb is an iterable containing numpy arrays of
        (mnb_N, n_vis) shape, where mnb_N is the number of
        samples in the minibatch.

    :param y_mnb: Trainset label indices split into minibatches. Thus,
        y_mnb is an iterable containing numpy arrays of
        (mnb_N, ) shape, where mnb_N is the number of
        samples in the minibatch.
    """

    #   gradients and param updates
    grads = [(p, theano.tensor.grad(cost=cost, wrt=p)) for p in params]
    updates = [(p, p - eps * grad_p) for (p, grad_p) in grads]

    # compiled training function
    train_model = theano.function(
        inputs=inputs,
        outputs=cost,
        updates=updates
    )

    #   things we'll track through training, for reporting
    epoch_costs = []
    epoch_times = []

    #   iterate through the epochs
    for epoch in range(epochs):
        log.info('Starting epoch %d', epoch)
        epoch_t0 = time()

        #   iterate through the minibatches
        batch_costs = []
        for batch_ind, (x_batch, y_batch) in enumerate(zip(x_mnb, y_mnb)):
            batch_costs.append(train_model(x_batch, y_batch))

        epoch_costs.append(np.array(batch_costs).mean())
        epoch_times.append(time() - epoch_t0)
        log.info(
            'Epoch cost %.5f, duration %.2f sec',
            epoch_costs[-1],
            epoch_times[-1]
        )

    log.info('Training duration %.2f min',
             (sum(epoch_times)) / 60.0)

    return epoch_costs, epoch_times


def write_ndarray(ndarray, file, formatter=None, separators=None):
    """
    Writes a numpy array into a file.

    :param ndarray: The array to write to file.
    :param file: File object in which to write.
    :param formatter: Formatting string to be used on each
        numpy array element if None (default), the '{}' is used.
    :param separators: A list of separator tokens to be used
        in between of array elements.
    """

    shape = ndarray.shape
    #   get cumulative sizes of each dimension
    dim_sizes = [
        np.prod(shape[(i + 1):], dtype=int) for i in range(0, len(shape))]

    #   prepare the separators
    if separators is None:
        separators = ['\n'] * len(shape)
        separators[-1] = ' '

    #   default formatter
    if formatter is None:
        formatter = "{}"

    #   write all the array elements
    for i, n in enumerate(ndarray.reshape(ndarray.size, )):
        if i != 0:
            sep_ind = [i % ds for ds in dim_sizes].index(0)
            file.write(separators[sep_ind])
        file.write(formatter.format(n))


def store_mlp_ascii(mlp, file_path):
    """
    Stores a MLP into an ASCII file.

    :param mlp: A MLP instance to store.
    :param file_path: File path to store it to.
    """

    log.info("Storing MLP to file: %s", file_path)

    #   first info in the ascii file is the layer sizes
    layer_sizes = [32 * 24]
    for hid_lay in mlp.hidden_layers:
        layer_sizes.append(hid_lay.b.get_value().size)
    layer_sizes.append(mlp.regression_layer.b.get_value().size)

    with open(file_path, "w") as file:

        def ln(string):
            file.write(string + '\n')

        ln("# Multilayer-perceptron, exported from Theano+Python DBN-MLP")
        ln("# Author: Florijan Stamenkovic (florijan.stameknovic@gmail.com")
        ln("# Date: {}".format(date.today()))
        ln("#")
        ln("# Non-comment lines are organized as follows:")
        ln("#   - first come layer sizes (visible -> hidden -> softmax")
        ln("#   - then for each layer (except visible):")
        ln("#       - first the weights to previous layer in N lines where N "
            "is number of neurons of previous layer")
        ln("#       - then biases for that layer (in a single line)")
        ln("# Enjoy!!!")

        file.write(" ".join([str(ls) for ls in layer_sizes]))

        for hl in mlp.hidden_layers:
            file.write('\n')
            write_ndarray(hl.W.get_value(), file, "{:.06f}")
            file.write('\n')
            write_ndarray(hl.b.get_value(), file, "{:.06f}")

        file.write('\n')
        write_ndarray(mlp.regression_layer.W.get_value(), file, "{:.06f}")
        file.write('\n')
        write_ndarray(mlp.regression_layer.b.get_value(), file, "{:.06f}")
        file.write('\n')
