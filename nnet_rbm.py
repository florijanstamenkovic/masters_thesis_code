import os
import sys
import util
import logging

import matplotlib.pyplot as plt
import theano
import numpy as np

import data
# from lrbm import LRBM
from llbl import LLBL
from llbl2 import LLBL2
from lmlp import LMLP
from ngram import NgramModel


#   dir where we store NNet models
_DIR = 'nnet_models'

#   how many ngrams from the validation set should be used
#   when evaluating exact log-likelihood... the whole validation
#   set can't be used because this is SLOW
_LL_SIZE = 500

#   logger
log = logging.getLogger(__name__)


def random_ngrams(ngrams, vocab_size, all=False, dist=None, shuffle=False):
    """
    Given an array of ngrams, creates a copy of that array
    with some terms (columns) randomized.

    :param ngrams: A numpy array of ngrams, of shape (N, n),
        where N is number of ngrams, and n is ngram length.
    :param vocab_size: Vocabulary size.
    :param all: If all ngram terms should be randomized, or just
        the conditioned one.
    :param shuffle: If randomization should be done by shuffling,
        or by sampling.
    :param dist: Probability distribution of words in the vocabulary.
        If None, uniform sampling is used.

    """
    r_val = np.array(ngrams)

    #   iterate through the terms that need replacing
    for term in xrange(ngrams.shape[1] if all else 1):

        #   iterate through the features of the term
        if shuffle:
            np.random.shuffle(r_val[:, term])
        else:
            r_val[:, term] = np.random.choice(
                vocab_size, ngrams.shape[0], p=dist).astype('uint16')

    return r_val


def main():
    """
    Trains and evaluates neural
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
        -a ALPHA : The alpha parameter of the model, defaults to 0.5
        -eps EPS : Learning rate, defaults to 0.005.
        -mnb MBN_SIZE : Size of the minibatch, defaults to 2000.

    """
    logging.basicConfig(level=logging.DEBUG)
    log.info("Evaluating model")

    #   get the data handling parameters
    ts_reduction = util.argv('-s', None, int)
    min_occ = util.argv('-o', 5, int)
    min_files = util.argv('-f', 2, int)
    n = util.argv('-n', 4, int)
    use_tree = '-t' in sys.argv
    bool_format = lambda s: s.lower() in ["1", "true", "yes", "t", "y"]
    ft_format = lambda s: map(bool_format, s)
    ftr_use = np.array(util.argv('-u', ft_format("001000"), ft_format))
    val_per_epoch = util.argv('-v', 10, int)

    #   nnets only support one-feature ngrams
    assert ftr_use.sum() == 1

    #   get nnet training parameters
    use_lbl = '-l' in sys.argv
    epochs = util.argv('-ep', 20, int)
    alpha = util.argv('-a', 0.5, float)
    eps = util.argv('-eps', 0.002, float)
    mnb_size = util.argv('-mnb', 2000, int)
    n_hid = util.argv('-h', 1000, int)
    d = util.argv('-d', 100, int)

    #   load data
    ngrams, q_groups, answers, feature_sizes = data.load_ngrams(
        n, ftr_use, use_tree, subset=ts_reduction,
        min_occ=min_occ, min_files=min_files)
    used_ftr_sizes = feature_sizes[ftr_use]
    #   remember, we only use one feature
    vocab_size = used_ftr_sizes[0]
    log.info("Data loaded, %d ngrams", ngrams.shape[0])

    #   split data into sets
    x_train, x_valid, x_test = util.dataset_split(ngrams, 0.05, 0.05, rng=456)

    #   generate a version of the validation set that has
    #   the first term (the conditioned one) randomized
    #   w.r.t. unigram distribution
    #   so first create the unigram distribution, no smoothing
    unigrams_data = data.load_ngrams(1, ftr_use, False, subset=ts_reduction,
                                     min_occ=min_occ, min_files=min_files)[0]
    unigrams_data = NgramModel(1, False, ftr_use, feature_sizes, ts_reduction,
                               min_occ, min_files, 0.0, 0.0, unigrams_data)
    unigrams_dist = unigrams_data.probability_additive(
        np.arange(vocab_size).reshape(vocab_size, 1))
    unigrams_dist /= unigrams_dist.sum()
    #   finally, generate validation sets with randomized term
    x_valid_r = random_ngrams(x_valid, vocab_size, False, unigrams_dist)

    #   the directory for this model
    dir = "%s_%s_%d-gram_features-%s_data-subset_%r-min_occ_%r-min_files_%r" % (
        "llbl" if use_lbl else "lmlp",
        "tree" if use_tree else "linear", n,
        "".join([str(int(b)) for b in ftr_use]),
        ts_reduction, min_occ, min_files)
    dir = os.path.join(_DIR, dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    #   filename base for this model
    file = "nhid-%d_d-%d_train_mnb-%d_epochs-%d_eps-%.5f_alpha-%.2f" % (
        n_hid, d, mnb_size, epochs, eps, alpha)

    #   store the logs
    if False:
        log_file_handler = logging.FileHandler(
            os.path.join(dir, file + ".log"))
        log_file_handler.setLevel(logging.INFO)
        logging.root.addHandler(log_file_handler)

    #   we will plot log-lik ratios for every _VALIDATE_MNB minibatches
    #   we will also plot true mean log-lik
    valid_on = {"x_valid": x_valid[:_LL_SIZE], "x_valid_r": x_valid_r[
        :_LL_SIZE], "x_train": x_train[:_LL_SIZE]}
    valid_ll = {k: [] for k in valid_on.keys()}
    valid_p_mean = {k: [] for k in valid_on.keys()}

    #   how often we validate
    mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
    _VALIDATE_MNB = mnb_count / val_per_epoch

    def mnb_callback(net, epoch, mnb):
        """
        Callback function called after every minibatch.
        """
        if (mnb + 1) % _VALIDATE_MNB:
            return

        #   calculate log likelihood using the exact probability
        probability_f = theano.function([net.input], net.probability)
        for name, valid_set in valid_on.iteritems():
            p = probability_f(valid_set)
            valid_ll[name].append(np.log(p).mean())
            valid_p_mean[name].append(p.mean())

        log.info('Epoch %d, mnb: %d, x_valid mean-log-lik: %.5f'
                 ' , x_valid p-mean: %.5f'
                 ' , ln(p(x_valid) / p(x_valid_r).mean(): %.5f',
                 epoch, mnb, valid_ll["x_valid"][-1],
                 valid_p_mean["x_valid"][-1],
                 valid_ll["x_valid"][-1] - valid_ll["x_valid_r"][-1])

    #   track if the model progresses on the sentence completion challenge
    # sent_challenge = []

    def epoch_callback(net, epoch):

        #   log some info about the parameters, just so we know
        param_mean_std = [(k, v.mean(), v.std())
                          for k, v in net.params().iteritems()]
        log.info("Epoch %d: %s", epoch, "".join(
            ["\n\t%s: %.5f +- %.5f" % pms for pms in param_mean_std]))

        #   evaluate model on the sentence completion challenge
        # probability_f = theano.function([net.input], net.probability)
        # qg_log_lik = [[np.log(probability_f(q)).sum() for q in q_g]
        #               for q_g in q_groups]
        # predictions = map(lambda q_g: np.argmax(q_g), qg_log_lik)
        # sent_challenge.append((np.array(predictions) == answers).mean())
        # log.info('Epoch %d sentence completion eval score: %.4f',
        #          epoch, sent_challenge[-1])

    log.info("Creating model")
    if use_lbl:
        net = LLBL2(n, vocab_size, d, 12345)
    else:
        net = LMLP(n, vocab_size, d, 12345)
    net.mnb_callback = mnb_callback
    net.epoch_callback = epoch_callback
    train_cost, valid_cost, _ = net.train(
        x_train, x_valid, mnb_size, epochs, eps, alpha)

    #   plot training progress info
    #   first we need values for the x-axis (minibatch count)
    mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
    mnb_valid_ep = mnb_count / _VALIDATE_MNB
    x_axis_mnb = np.tile((np.arange(mnb_valid_ep) + 1) * _VALIDATE_MNB, epochs)
    x_axis_mnb += np.repeat(np.arange(epochs) * mnb_count, mnb_valid_ep)
    x_axis_mnb = np.hstack(([0], x_axis_mnb))

    plt.figure(figsize=(16, 12))
    plt.subplot(221)
    plt.plot(mnb_count * (np.arange(epochs) + 1), train_cost, 'b-',
             label='train')
    plt.plot(mnb_count * (np.arange(epochs) + 1), valid_cost, 'g-',
             label='valid')
    plt.title('cost')
    plt.grid()
    plt.legend(loc=1)

    plt.subplot(222)
    for name, valid_set in valid_ll.items():
        plt.plot(x_axis_mnb, valid_set, label=name)
    plt.ylim((np.log(0.5 / vocab_size),
              max([max(v) for v in valid_ll.values()]) + 0.5))
    plt.title('log-likelihood(x)')
    plt.grid()
    plt.legend(loc=4)

    plt.subplot(224)
    for name, valid_set in valid_p_mean.items():
        plt.plot(x_axis_mnb, valid_set, label=name)
    plt.title('p(x).mean()')
    plt.grid()
    plt.legend(loc=4)

    # plt.subplot(224)
    # plt.plot(mnb_count * np.arange(epochs + 1), sent_challenge, 'g-')
    # plt.title('sent_challenge')
    # plt.grid()

    plt.savefig(os.path.join(dir, file + ".pdf"))


if __name__ == '__main__':
    main()
