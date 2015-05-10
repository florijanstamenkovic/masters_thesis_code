"""
Final model evaluation.
Compares ngrams (with optimized additive smoothing),
an LRBM and LLBL.
"""

import logging
import sys
import os

import numpy as np
import theano
import matplotlib.pyplot as plt

import ngram
import util
import data
from llbl import LLBL

log = logging.getLogger(__name__)

#   we won't validate nets only after each epoch (large dataset)
#   but after each _VALIDATE_MNB minibatches
_VALIDATE_MNB = 25


_DIR = 'eval'
if not os.path.exists(_DIR):
    os.makedirs(_DIR)


def main():
    logging.basicConfig(level=logging.INFO)
    log.info("Performing final eval")

    #   get the data handling parameters
    ts_reduction = util.argv('-s', None, int)
    min_occ = util.argv('-o', 5, int)
    min_files = util.argv('-f', 2, int)
    n = util.argv('-n', 4, int)
    use_tree = '-t' in sys.argv
    bool_format = lambda s: s.lower() in ["1", "true", "yes", "t", "y"]
    ft_format = lambda s: map(bool_format, s)
    ftr_use = np.array(util.argv('-u', ft_format("001000"), ft_format))

    #   nnet rbm-s only support one-feature ngrams
    assert ftr_use.sum() == 1

    #   get nnet training parameters
    epochs = util.argv('-ep', 20, int)
    alpha = util.argv('-a', 0.5, float)
    eps_llbl = util.argv('-eps_llbl', 0.1, float)
    eps_lrbm = util.argv('-eps_lrbm', 0.01, float)
    mnb_size = util.argv('-mnb', 500, int)
    n_hid = util.argv('-h', 1000, int)
    d = util.argv('-d', 150, int)

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

    def eval_ngram():
        """
        A function that creates and evaluates ngram models
        with additive smooting.
        """
        #   get and eval Ngram model
        ngram_model = ngram.NgramModel(
            n, use_tree, ftr_use, feature_sizes,
            ts_reduction, min_occ, min_files, 0.0, x_train)

        #   optimize lambda on the validation set
        ngram_lmbd = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

        def perplexity(lmbd, valid_set=True):
            ngram_model.lmbd = lmbd
            dataset = x_valid if valid_set else x_test
            probability = ngram_model.probability(dataset)
            log_loss = -np.log(probability).mean()
            perplexity = np.exp(log_loss)
            log.info("Ngrams, lmbd=%.5f: log_loss: %.2f, perplexity: %.2f",
                     lmbd, log_loss, perplexity)
            return perplexity

        best_lmbd = ngram_lmbd[np.argmin(map(perplexity, ngram_lmbd))]
        log.info("Final eval of ngram model:")
        perplexity(best_lmbd, False)

    def eval_llbl():
        """
        A function that creates, trains and evaluates LLBL
        models with additive smoothing.
        """
        #   create and evaluate a LLBL
        net = LLBL(n, vocab_size, d, 64353)
        #   let's track how smoothing influences validation cost
        llbl_lmbd = [0., 1e-6, 1e-4]
        llbl_logloss = dict(zip(
            llbl_lmbd, [[] for i in range(len(llbl_lmbd))]))

        partition_exp_f = theano.function([net.input], net.partition_exp)

        def mnb_callback(llbl, epoch, mnb):
            if (mnb + 1) % _VALIDATE_MNB:
                return

            _probs = map(partition_exp_f, util.create_minibatches(
                x_valid, None, mnb_size, False))
            _probs = np.vstack(_probs)

            for lmbd in llbl_lmbd:
                probs = _probs + lmbd
                probs /= np.expand_dims(probs.sum(axis=1), axis=1)
                probs = probs[np.arange(x_valid.shape[0]), x_valid[:, 0]]
                log_loss = -np.log(probs).mean()
                perplexity = np.exp(log_loss)
                log.info("LLBL, epoch %d, mnb %d, lmbd=%.6f:"
                         " log_loss: %.2f, perplexity: %.2f",
                         epoch, mnb, lmbd, log_loss, perplexity)
                llbl_logloss[lmbd].append(log_loss)

        net.mnb_callback = mnb_callback
        train_cost, valid_cost, _ = net.train(
            x_train, x_valid, mnb_size, epochs, eps_llbl, alpha)

        #   plot training progress info
        #   first we need values for the x-axis (minibatch count)
        mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
        mnb_valid_ep = mnb_count / _VALIDATE_MNB
        x_axis_mnb = np.tile(
            (np.arange(mnb_valid_ep) + 1) * _VALIDATE_MNB, epochs)
        x_axis_mnb += np.repeat(np.arange(epochs) * mnb_count, mnb_valid_ep)
        x_axis_mnb = np.hstack(([0], x_axis_mnb))
        #   now plot the log losses
        plt.figure(figsize=(16, 12))
        for lmbd, scores in llbl_logloss.iteritems():
            plt.plot(x_axis_mnb, scores, label='lmbd=%.5f' % lmbd)
        plt.title('LLBL validation log-loss')
        plt.grid()
        plt.legend()

        plt.savefig(os.path.join(_DIR, "llbl_validation.pdf"))

    eval_llbl()


if __name__ == '__main__':
    main()
