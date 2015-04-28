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


def random_ngrams(ngrams, feature_sizes, terms=1, shuffle=False,
                  distributions=None):
    """
    Given an array of ngrams, creates a copy of that array
    with some terms (columns) randomized.

    :param ngrams: A numpy array of ngrams, of shape (N, n),
        where N is number of ngrams, and n is ngram length.
    :param feature_sizes: Vocabulary size for features used
        in each ngram term.
    :param terms: How many terms in the ngram should be randomized.
    :param shuffle: If randomization should be done by shuffling,
        or by sampling.
    :param distributions:
    """

    if distributions is None:
        distributions = [None] * len(feature_sizes)

    #   randomized ngrams
    r_val = np.array(ngrams)

    #   iterate through the terms that need replacing
    for term in xrange(terms):

        #   iterate through the features of the term
        for ftr_ind, (feature_size, dist) in enumerate(
                zip(feature_sizes, distributions)):

            ftr_ind += term * len(feature_sizes)

            if shuffle:
                np.random.shuffle(r_val[:, ftr_ind])
            else:
                r_val[:, ftr_ind] = np.random.choice(
                    feature_size, ngrams.shape[0],
                    p=dist).astype('uint16')

    return r_val


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

    #   the directory that stores ngram models we compare against
    ngram_dir = NgramModel.dir(use_tree, ftr_use, ts_reduction,
                               min_occ, min_files)

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

    #   generate a version of the validation set that has
    #   the first term (the conditioned one) randomized
    #   w.r.t. unigram distribution
    unigrams_data = data.load_ngrams(1, ftr_use, False, subset=ts_reduction,
                                     min_occ=min_occ, min_files=min_files)[0]
    #   generate unigram distributions for each feature individually
    used_ftrs_one_hot = [
        x for x in np.eye(len(ftr_use), dtype=bool) if x[ftr_use].any()]
    unigrams_dist = [NgramModel.get(1, False, ftr, feature_sizes,
                                    unigrams_data, ngram_dir, lmbd=0.0) for
                     ftr in used_ftrs_one_hot]
    unigrams_dist = [m.probability(np.arange(l).reshape(l, 1)) for
                     m, l in zip(unigrams_dist, used_ftr_sizes)]
    for ud in unigrams_dist:
        ud /= ud.sum()
    #   finally, generate a validation set with randomized conditioned term
    x_valid_r = random_ngrams(x_valid, used_ftr_sizes,
                              distributions=unigrams_dist)
    #   also generate a validation set with all terms randomized
    x_valid_rr = random_ngrams(x_valid, used_ftr_sizes, n,
                               distributions=unigrams_dist)
    x_valid_rrr = random_ngrams(x_valid, used_ftr_sizes, n)

    #   the directory for this model
    dir = "%s_%d-gram_features-%s_data-subset_%r-min_occ_%r-min_files_%r" % (
        "tree" if use_tree else "linear", n,
        "".join([str(int(b)) for b in ftr_use]),
        ts_reduction, min_occ, min_files)
    dir = os.path.join(_DIR, dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    #   filename base for this model
    file = "nhid-%d_d-%d_train_mnb-%d_epochs-%d_eps-%.5f_alpha-%.1f" % (
        n_hid, d, mnb, epochs, eps, alpha)

    #   store the logs
    log_file_handler = logging.FileHandler(os.path.join(dir, file + ".log"))
    log_file_handler.setLevel(logging.INFO)
    logging.root.addHandler(log_file_handler)

    #   default vector representation sizes
    repr_sizes = np.array([d, d, d, 10, 10, 10], dtype='uint8')

    #   get the ngram probability for the validation set
    #   first load the relevant ngram model
    ngram_model = NgramModel.get(n, use_tree, ftr_use, feature_sizes,
                                 x_train, ngram_dir)
    #   then calculate validation set probability
    #   and also the probability of validation set with randomized
    #   conditioned term
    p_ngram = ngram_model.probability(x_valid)
    p_ngram_r = ngram_model.probability(x_valid_r)
    p_ngram_rr = ngram_model.probability(x_valid_rr)
    p_ngram_rrr = ngram_model.probability(x_valid_rrr)
    log.info("Ngram model ln(p_mean(x_valid)): %.3f, ln(p_mean(x_valid_r)):"
             " %3f, ln(p_mean(x_valid_rr)): %3f, ln(p_mean(x_valid_rrr)): %3f",
             np.log(p_ngram.mean()), np.log(p_ngram_r.mean()),
             np.log(p_ngram_rr.mean()), np.log(p_ngram_rrr.mean()))
    log.info("Ngram model p(x_valid) / p(x_valid_rand) mean: %.3f, "
             "p(x_valid) / p(rand) mean: %.3f,",
             (p_ngram / p_ngram_r).mean(),
             (p_ngram / p_ngram_rr).mean())
    log.info("Ngram model ll_mean(x_valid): %.3f, ll_mean(x_valid_r): %3f, "
             "ll_mean(x_valid_rr): %3f, ll_mean(x_valid_rrr): %3f",
             np.log(p_ngram).mean(), np.log(p_ngram_r).mean(),
             np.log(p_ngram_rr).mean(), np.log(p_ngram_rrr).mean())
    log.info("Ngram model ll(x_valid) / ll("
             "x_valid_rand) mean: %.3f, ll(x_valid) / ll(rand) mean: %.3f,",
             (np.log(p_ngram) - np.log(p_ngram_r)).mean(),
             (np.log(p_ngram) - np.log(p_ngram_rr)).mean())

    def validation_energy(lrbm):
        """
        For the given RBM this function calculates the energies
        of the validation set and it's randomized versions.

        Returns a tuple: (
            validation set energy,
            randomized (one term) validation set energy,
            randomized (all terms) validation set energy,
            log-likelihood ratio of validation and randomized (one term) sets
        )
        """
        energy_f = theano.function([lrbm.input], lrbm.energy)

        x_valid_en = energy_f(x_valid)
        x_valid_en_r = energy_f(x_valid_r)
        x_valid_en_rr = energy_f(x_valid_rr)

        return (x_valid_en.mean(), x_valid_en_r.mean(), x_valid_en_rr.mean(),
                (-x_valid_en + x_valid_en_r).mean())

    #   we will plot log-lik ratios for every 5 minibatches
    #   we will also plot true mean log-lik
    llratios_mnb = []
    llmean_mnb = []
    llmean_r_mnb = []

    def mnb_callback(lrbm, epoch, mnb):
        """
        Callback function called after every minibatch.
        """
        if mnb == 0 or mnb % 5:
            return
        validation_energies = validation_energy(lrbm)
        llratios_mnb.append(validation_energies[-1])
        log.info(
            'Epoch %d, mnb: %d, x_valid_en: %.2f, x_valid_en_r: %.2f, '
            'x_valid_en_rr: %.2f, [-x_valid_en + x_valid_en_r].mean(): %.3f',
            epoch, mnb, *validation_energies)

        llmean_mnb.append(lrbm.mean_log_lik(x_valid[:50]))
        llmean_r_mnb.append(lrbm.mean_log_lik(x_valid_r[:50]))
        log.info('Epoch %d, mnb: %d, x_valid mean-log-lik: %.5f, '
                 'x_valid_r mean-log-lik: %.5f',
                 epoch, mnb, llmean_mnb[-1], llmean_r_mnb[-1])

    def epoch_callback(lrbm, epoch):

        #   we'll use the net's energy function to eval q_groups
        energy_f = theano.function([lrbm.input], lrbm.energy)

        #   get the net energy on the validation set and a randomized set
        log.info(
            'Epoch %d, x_valid_en: %.2f, x_valid_en_r: %.2f, '
            'x_valid_en_rr: %.2f, [-x_valid_en + x_valid_en_r].mean(): %.3f',
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

    #   plot llratios and llmean
    fig, ax1 = plt.subplots()
    x = (np.arange(len(llratios_mnb))) * 5

    ax2 = ax1.twinx()
    ax1.plot(x, llratios_mnb, 'g-')
    ax1.set_ylabel('log-lik ratio', color='g')
    ax2.plot(x, llmean_mnb, 'b-', label='ll(valid_x)')
    ax2.plot(x, llmean_r_mnb, 'b--', label='ll(valid_x_r)')
    ax2.set_ylabel('log-lik mean', color='b')
    ax1.set_xlabel('minibatch')
    plt.grid()
    plt.legend(loc=2)
    plt.savefig(file + ".pdf")


if __name__ == '__main__':
    main()
