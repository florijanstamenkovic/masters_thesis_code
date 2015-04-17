"""
Module for evaluating the RBM energy-based neural net
language models on the Microsoft
Sentences Completion Challenge dataset (obtained through
the 'data' module).
"""

import os
import sys
import util
import data
import logging
import numpy as np
from lrbm import LRBM
import theano

#   dir where we store NNet models
_DIR = 'nnet_models'

log = logging.getLogger(__name__)


def dataset_split(x, validation=0.05, test=0.05, rng=None):
    """
    Splits dataset into train, validation and testing subsets.
    The dataset is split on the zeroth axis.

    :param x: The dataset of shape (N, ...)
    :param validation: float in range (0, 1) that indicates
        desired validation set size to be N * validation
    :param test: float in range (0, 1) that indicates
        desired test set size to be N * test
    :param rng: Numpy random number generator, or an integer
        seed for rng, or None (rng initialized always with the same seed).
    """
    assert validation > 0. and test > 0.
    assert validation + test < 1.

    log.info("Performing dataset split, validation size: %.2f, "
             "test size: %.2f", validation, test)

    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)

    #   shuffle data
    rng.shuffle(x)

    #   generate split indices
    i1 = int(x.shape[0] * (1. - validation - test))
    i2 = int(x.shape[0] * (1. - test))

    return x[:i1], x[i1:i2], x[i2:]


def main():
    """
    Trains and evaluates RBM energy based neural net
    language models on the Microsoft Sentence Completion
    Challenge dataset.

    Allowed cmd-line flags:
        -s TS_FILES : Uses the reduced trainsed (TS_FILES trainset files)
        -o MIN_OCCUR : Only uses terms that occur MIN_OCCUR or more times
            in the trainset. Other terms are replaced with a special token.
        -f MIN_FILES : Only uses terms that occur in MIN_FILES or more files
            in the trainset. Other terms are replaced with a special token.
        -n : n-gram length (default 4)
        -t : Use tree-grams (default does not ues tree-grams)
        -u FTRS : Features to use. FTRS must be a string composed of zeros
            and ones, of length 5. Ones indicate usage of following features:
            (word, lemma, google_pos, penn_pos, dependency_type), respectively.

    Neural-net specific cmd-line flags:
        -ep EPOCHS : Number of training epochs, defaults to 20.
        -a ALPHA : The alpha parameter of the LRBM, defaults to 0.5
        -eps EPS : Learning rate, defaults to 0.005.
        -mnb MBN_SIZE : Size of the minibatch, defaults to 2000.

    """
    logging.basicConfig(level=logging.DEBUG)
    log.info("RBM energy-based neural net language model")

    #   get the data handling parameters
    ts_reduction = util.argv('-s', None, int)
    min_occ = util.argv('-o', 5, int)
    min_files = util.argv('-f', 2, int)
    n = util.argv('-n', 4)
    use_tree = '-t' in sys.argv
    ft_format = lambda s: map(
        lambda s: s.lower() in ["1", "true", "yes", "t", "y"], s)
    ftr_use = np.array(util.argv('-u', ft_format("001000"), ft_format))

    #   get nnet training parameters
    epochs = util.argv('-ep', 20, int)
    alpha = util.argv('-a', 0.5, float)
    eps = util.argv('-eps', 0.005, float)
    mnb = util.argv('-mnb', 2000, int)
    n_hid = util.argv('-h', 1000, int)
    d = util.argv('-d', 100, int)

    #   load data
    ngrams, q_groups, answers, feature_sizes = data.load_ngrams(
        n, ftr_use, use_tree, subset=ts_reduction,
        min_occ=min_occ, min_files=min_files)
    log.info("Data loaded, %d ngrams", ngrams.shape[0])

    #   split data into sets
    x_train, x_valid, x_test = dataset_split(ngrams, 0.05, 0.05, rng=12345)

    #   the directory for this model
    dir = "%s_%d-gram_features-%s_data-subset_%r-min_occ_%r-min_files_%r" % (
        "tree" if use_tree else "linear", n,
        "".join([str(int(b)) for b in ftr_use]),
        ts_reduction, min_occ, min_files)
    dir = os.path.join(_DIR, dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    #   filename base for this model
    file = "nhid-%d_d-%d_train_mnb-%d_epochs-%d_eps-%.3f_alpha-%.1f" % (
        n_hid, d, mnb, epochs, eps, alpha)

    #   store the logs
    log_file_handler = logging.FileHandler(os.path.join(dir, file + ".log"))
    log_file_handler.setLevel(logging.INFO)
    logging.root.addHandler(log_file_handler)

    #   default vector representation sizes
    repr_sizes = np.array([d, d, d, 10, 10, 10], dtype='uint8')

    #   epoch callback used for evaluation on the data completion challenge
    def epoch_callback(lrbm, epoch):

        #   we'll use the net's energy function to eval q_groups
        energy_f = theano.function([lrbm.input], lrbm.energy)
        qg_energies = map(
            lambda q_g: [energy_f(q).sum() for q in q_g], q_groups)
        predictions = map(lambda q_g: np.argmax(q_g), qg_energies)
        log.info(
            'Epoch %d sentence completion eval score: %.4f, energy: %.2f',
            epoch,
            (np.array(predictions) == answers).sum() / float(answers.size),
            np.sum(map(sum, qg_energies))
        )

    log.info("Creating LRBM")
    lrbm = LRBM(n, feature_sizes[ftr_use], repr_sizes[ftr_use], n_hid, 12345)
    lrbm.epoch_callback = epoch_callback
    lrbm.train(x_train, x_valid, mnb, epochs, eps, alpha)

if __name__ == '__main__':
    main()
