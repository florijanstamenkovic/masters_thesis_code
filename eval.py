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
from lrbm import LRBM

log = logging.getLogger(__name__)


_DIR = 'eval'
if not os.path.exists(_DIR):
    os.makedirs(_DIR)


def main():
    logging.basicConfig(level=logging.DEBUG)
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
    val_per_epoch = util.argv('-v', 10, int)

    #   nnet rbm-s only support one-feature ngrams
    assert ftr_use.sum() == 1

    #   get nnet training parameters
    epochs = util.argv('-ep', 20, int)
    alpha = util.argv('-a', 0.5, float)
    eps_llbl = util.argv('-eps_llbl', 0.05, float)
    eps_lrbm = util.argv('-eps_lrbm', 0.005, float)
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
    x_train, x_valid, x_test = util.dataset_split(
        ngrams, int(min(1e4, 0.05 * ngrams.shape[0])), 0.05, rng=456)

    #   how often we validate
    mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
    _VALIDATE_MNB = mnb_count / val_per_epoch

    def eval_ngram(smoothing_param, use_kn=True):
        """
        A function that creates and evaluates ngram models
        with [additive / knesser-ney] smooting.
        """
        #   get and eval Ngram model
        ngram_model = ngram.NgramModel(
            n, use_tree, ftr_use, feature_sizes,
            ts_reduction, min_occ, min_files, 0.0, 0.0, x_train)

        def perplexity(smoothing, valid_set=True):
            dataset = x_valid if valid_set else x_test
            if use_kn:
                ngram_model.delta = smoothing
                probability = ngram_model.probability_kn(dataset)
            else:
                ngram_model.lmbd = smoothing
                probability = ngram_model.probability_additive(dataset)

            log_loss = -np.log(probability).mean()
            perplexity = np.exp(log_loss)
            log.info("Ngrams %s, smoothing=%.e: log_loss: %.4f, perplexity: %.2f",
                     "knesser_ney" if use_kn else "additive",
                     smoothing, log_loss, perplexity)
            return perplexity

        best_lmbd = ngram_lmbd[np.argmin(map(perplexity, ngram_lmbd))]
        log.info("Final eval of ngram model:")
        perplexity(best_lmbd, False)

    def eval_net(use_llbl=True, lmbds=[0.]):
        """
        A function that creates, trains and evaluates an LRBM or LLBLs
        models with additive smoothing.
        """
        #   create and evaluate a LLBL
        if use_llbl:
            net = LLBL(n, vocab_size, d, 64353)
        else:
            net = LRBM(n, vocab_size, d, n_hid, 64353)

        #   let's track how smoothing influences validation cost
        lmbds_log_loss = dict(zip(lmbds, [[] for i in range(len(lmbds))]))

        distr_w_unn = theano.function([net.input], net.distr_w_unn)

        def mnb_callback(net, epoch, mnb):
            if (max(epoch, 0) * mnb_count + mnb + 1) % _VALIDATE_MNB:
                return

            _probs = map(distr_w_unn, util.create_minibatches(
                x_valid, None, mnb_size, False))
            _probs = np.vstack(_probs)

            for lmbd in lmbds:
                probs = _probs + lmbd
                probs /= np.expand_dims(probs.sum(axis=1), axis=1)
                probs = probs[np.arange(x_valid.shape[0]), x_valid[:, 0]]
                log.debug('Probs mean: %.6f', probs.mean())
                log_loss = -np.log(probs).mean()
                perplexity = np.exp(log_loss)
                log.info("%s, epoch %d, mnb %d, lmbd=%.e:"
                         " log_loss: %.4f, perplexity: %.2f",
                         "LLBL" if use_llbl else "LRBM",
                         epoch, mnb, lmbd, log_loss, perplexity)
                lmbds_log_loss[lmbd].append(log_loss)

        net.mnb_callback = mnb_callback
        train_cost, valid_cost, _ = net.train(
            x_train, x_valid, mnb_size, epochs,
            eps_llbl if use_llbl else eps_lrbm, alpha)

        #   plot training progress info
        #   first we need values for the x-axis (minibatch count)
        mnb_valid_ep = mnb_count / _VALIDATE_MNB
        x_axis_mnb = np.tile(
            (np.arange(mnb_valid_ep) + 1) * _VALIDATE_MNB, epochs)
        x_axis_mnb += np.repeat(np.arange(epochs) * mnb_count, mnb_valid_ep)
        x_axis_mnb = np.hstack(([0], x_axis_mnb))
        #   now plot the log losses
        plt.figure(figsize=(16, 12))
        for lmbd, scores in lmbds_log_loss.iteritems():
            plt.plot(x_axis_mnb, scores, label='lmbd=%.e' % lmbd)
        plt.title('%s validation log-loss' % "LLBL" if use_llbl else "LRBM")
        plt.grid()
        plt.legend()

        plt.savefig(os.path.join(_DIR, "llbl_validation.pdf"))

    #   evaluate ngram models, additive and knesser-ney
    ngram_lmbd = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    eval_ngram(ngram_lmbd, False)
    ngram_delta = [0.25, 0.5, 0.75, 1.0]
    eval_ngram(ngram_delta, True)

    # eval_net(True, [0., 1e-20, 1e-10])
    # eval_net(False, [0., 1e-20, 1e-10])


if __name__ == '__main__':
    main()
