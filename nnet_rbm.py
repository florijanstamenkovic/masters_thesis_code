"""
Module for evaluating the RBM energy-based neural net
language models on the Microsoft
Sentences Completion Challenge dataset (obtained through
the 'data' module).
"""

import os
import sys
import util
import logging

import matplotlib.pyplot as plt
import theano
import numpy as np

import data
from lrbm import LRBM
from ngram import NgramModel


#   dir where we store NNet models
_DIR = 'nnet_models'

log = logging.getLogger(__name__)


def random_ngrams(ngrams, feature_sizes, terms=1, shuffle=True):

    n = ngrams.shape[1] / len(feature_sizes)

    random_ngrams = np.array(ngrams)
    for term in xrange(terms):
        for feature, feature_size in enumerate(feature_sizes):
            ftr_ind = (n - 1 - term) * len(feature_sizes)
            if shuffle:
                np.random.shuffle(random_ngrams[:, ftr_ind])
            else:
                random_ngrams[:, ftr_ind] = np.random.randint(
                    0, feature_sizes, ngrams.shape[0]).astype('uint16')

    return random_ngrams


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
    eps = util.argv('-eps', 0.002, float)
    mnb = util.argv('-mnb', 2000, int)
    n_hid = util.argv('-h', 1000, int)
    d = util.argv('-d', 100, int)

    #   load data
    ngrams, q_groups, answers, feature_sizes = data.load_ngrams(
        n, ftr_use, use_tree, subset=ts_reduction,
        min_occ=min_occ, min_files=min_files)
    used_ftr_sizes = feature_sizes[ftr_use]
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

    #   get the ngram probability for the validation set
    ngram_dir = NgramModel.dir(
        use_tree, ftr_use, ts_reduction, min_occ, min_files)
    ngram_model = NgramModel.get(
        n, use_tree, ftr_use, feature_sizes, x_train, ngram_dir)
    p_ngram = ngram_model.probability(x_valid)
    p_ngram_rand = ngram_model.probability(
        random_ngrams(x_valid, used_ftr_sizes))
    p_rand = ngram_model.probability(
        random_ngrams(x_valid, used_ftr_sizes, n))
    log.info("Ngram model p(x_valid) / p(x_valid_rand) mean: %.3f, "
             "p(x_valid) / p(rand) mean: %.3f,",
             (p_ngram / p_ngram_rand).mean(),
             (p_ngram / p_rand).mean())
    log.info("Ngram model ll(x_valid) / ll(x_valid_rand) mean: %.3f, "
             "ll(x_valid) / ll(rand) mean: %.3f,",
             (np.log(p_ngram) - np.log(p_ngram_rand)).mean(),
             (np.log(p_ngram) - np.log(p_rand)).mean())

    def validation_energy(lrbm):

        energy_f = theano.function([lrbm.input], lrbm.energy)

        x_valid_en = energy_f(x_valid)
        x_valid_rand_en = energy_f(random_ngrams(x_valid, used_ftr_sizes))
        rand_en = energy_f(random_ngrams(x_valid, used_ftr_sizes, n))

        return (x_valid_en.mean(), x_valid_rand_en.mean(), rand_en.mean(),
                (-x_valid_en + x_valid_rand_en).mean())

    llratios_mnb = []

    def mnb_callback(lrbm, epoch, mnb):
        if mnb == 0 or mnb % 5:
            return
        validation_energies = validation_energy(lrbm)
        llratios_mnb.append(validation_energies[-1])
        log.info(
            'Epoch %d, mnb: %d, x_valid energy: %.2f, x_valid_rand energy: %.2f'
            ', rand energy: %.2f, [-x_valid_en + x_valid_rand].mean(): %.3f',
            epoch, mnb, *validation_energies)

    def epoch_callback(lrbm, epoch):

        #   we'll use the net's energy function to eval q_groups
        energy_f = theano.function([lrbm.input], lrbm.energy)

        #   get the net energy on the validation set and a randomized set
        log.info(
            'Epoch %d x_valid energy: %.2f, x_valid_rand energy: %.2f'
            ', rand energy: %.2f, [-x_valid_en + x_valid_rand].mean(): %.3f',
            epoch, *validation_energy(lrbm))

        #   log some info about parameters
        log.info("Epoch %d:\n\tw: %.5f +- %.5f\n\tb_hid: %.5f +- %.5f"
                 "\n\tb_vis: %.5f +- %.5f\n\trepr: %.5f +- %.5f",
                 epoch,
                 lrbm.w.get_value(borrow=True).mean(),
                 lrbm.w.get_value(borrow=True).std(),
                 lrbm.b_hid.get_value(borrow=True).mean(),
                 lrbm.b_hid.get_value(borrow=True).std(),
                 lrbm.b_vis.get_value(borrow=True).mean(),
                 lrbm.b_vis.get_value(borrow=True).std(),
                 np.mean([emb.get_value(borrow=True).mean()
                          for emb in lrbm.embeddings]),
                 np.mean([emb.get_value(borrow=True).std()
                          for emb in lrbm.embeddings])
                 )

        #   log info about the sentence completion challenge
        qg_energies = map(
            lambda q_g: [energy_f(q).sum() / max(1, q.shape[0])
                         for q in q_g], q_groups)
        predictions = map(lambda q_g: np.argmin(q_g), qg_energies)
        log.info(
            'Epoch %d sentence completion eval score: %.4f, energy: %.2f',
            epoch,
            (np.array(predictions) == answers).mean(),
            np.array(qg_energies).mean()
        )

    log.info("Creating LRBM")
    lrbm = LRBM(n, used_ftr_sizes, repr_sizes[ftr_use], n_hid, 12345)
    lrbm.mnb_callback = mnb_callback
    lrbm.epoch_callback = epoch_callback
    lrbm.train(x_train, x_valid, mnb, epochs, eps, alpha)

    #   plot llratios
    plt.figure(figsize=(12, 9), dpi=72)
    plt.plot(np.arange(len(llratios_mnb)) * 5 + 1, llratios_mnb)
    plt.xlabel("Minibatch")
    plt.ylabel("log-lik ratio")
    plt.grid()
    plt.savefig(file + ".pdf")


if __name__ == '__main__':
    main()
