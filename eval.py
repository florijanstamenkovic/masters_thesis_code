"""
Final model evaluation.
Compares ngrams (with optimized additive smoothing),
an LMLP and LLBL.
"""

import logging
import sys
import os
from time import time
import json

import numpy as np
import matplotlib.pyplot as plt
import theano

import ngram
import util
import data
from llbl import LLBL
from lnnet import LNNet

log = logging.getLogger(__name__)


def plot_log_loss(validation_lloss, training_lloss, path=None):
    #   plot training progress info
    #   now plot the log losses
    plt.figure(figsize=(16, 12))
    plt.plot(np.arange(len(validation_lloss)) + 1, validation_lloss,
             label='validation', color='g')
    plt.plot(np.arange(len(training_lloss)) + 1, training_lloss,
             label='train', color='b')
    plt.axhline(min(validation_lloss), linestyle='--', color='g')
    plt.yticks(list(plt.yticks()[0]) + [min(validation_lloss)])
    plt.ylabel("Log-loss")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


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
    use_lbl = '-l' in sys.argv

    #   nnet rbm-s only support one-feature ngrams
    assert ftr_use.sum() == 1

    #   get nnet training parameters
    epochs = util.argv('-ep', 20, int)
    eps_llbl = util.argv('-eps_llbl', 0.0001, float)
    eps_lmlp = util.argv('-eps_lmlp', 0.0001, float)
    mnb_size = util.argv('-mnb', 500, int)
    d = util.argv('-d', 150, int)

    def path():
        """
        Returns a file-name base (without extension)
        for the model being evaluated.
        """
        #   the directory for this model
        dir = "%s_%s_%d-gram_features-%s_data-subset_%r-min_occ_%r-min_files_%r"\
            % ("llbl" if use_lbl else "lmlp",
                "tree" if use_tree else "linear", n,
                "".join([str(int(b)) for b in ftr_use]),
                ts_reduction, min_occ, min_files)
        dir = os.path.join('eval', dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        #   filename base for this model
        eps = eps_llbl if use_lbl else eps_lmlp
        file = "d-%d_train_mnb-%d_epochs-%d_eps-%.5f" % (
            d, mnb_size, epochs, eps)

        return os.path.join(dir, file)

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

    def eval_msscc(prob_function):
        """
        Evaluates the given probability function on the
        Microsoft Research Sentence Completion Challenge.

        :param prob_function: A function that takes an array
            of ngrams of shape (N, n) and returns probabilities
            of each ngram.
        """
        sentence_prob = lambda sent: np.log(prob_function(sent)).sum()
        predictions = map(lambda qg: np.argmax(map(sentence_prob, qg)),
                          q_groups)
        return (answers == np.array(predictions)).mean()

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
                ngram_model.set_delta(smoothing)
                probability = ngram_model.probability_kn(dataset)
            else:
                ngram_model.set_lmbd(smoothing)
                probability = ngram_model.probability_additive(dataset)

            log_loss = -np.log(probability).mean()
            perplexity = np.exp(log_loss)
            log.info(
                "Ngrams %s, smoothing=%.e: log_loss: %.4f, perplexity: %.2f",
                "knesser_ney" if use_kn else "additive",
                smoothing, log_loss, perplexity)
            return perplexity

        best_lmbd = smoothing_param[
            np.argmin(map(perplexity, smoothing_param))]
        log.info("Final eval of ngram model: %.2f",
                 perplexity(best_lmbd, False))

        log.info("Final eval on MRSCC: %.3f",
                 eval_msscc(ngram_model.probability_kn))

    def eval_net():
        """
        A function that creates, trains and evaluates an LMLP or LLBL.
        """
        #   well store the results in a dictionary
        #   and return that
        results = {}

        #   create and evaluate a LLBL
        if use_lbl:
            net = LLBL(n, vocab_size, d, 64353)
        else:
            net = LNNet(n, vocab_size, d, 64353)

        #   train models while validation llos falls for a
        #   significant margin w.r.t. an epoch of training
        #   or a maximum number of epochs is reached
        epoch_llos = []
        results["epoch_validation_logloss"] = epoch_llos

        def epoch_callback(net, epoch):

            epoch_llos.append(net.evaluate(x_valid, mnb_size))
            log.info("Epoch %d, validation cost: %.4f", epoch, epoch_llos[-1])
            if epoch < 4:
                return True
            else:
                return np.exp(epoch_llos[-4]) - np.exp(epoch_llos[-1]) > 1.

        training_time = time()
        net.epoch_callback = epoch_callback
        train_cost = net.train(
            x_train, mnb_size, epochs,
            eps_llbl if use_lbl else eps_lmlp)
        training_time = time() - training_time
        results["train_time"] = training_time
        results["epoch_train_logloss"] = train_cost

        #   final evaluation on the test set
        evaluation = net.evaluate(x_test, mnb_size)
        results["final_eval"] = evaluation
        log.info("### Final evaluation score: %.4f", evaluation)

        #   final evaluation on the sentence completion
        prob_f = theano.function([net.input], net.probability)
        mrscc_eval = eval_msscc(prob_f)
        results["mrscc_eval"] = mrscc_eval
        log.info("### MRSCC evaluation score: %.3f", mrscc_eval)

        return results

    if '-eval_ngram' in sys.argv:
        #   evaluate ngram models, additive and knesser-ney
        ngram_lmbd = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        eval_ngram(ngram_lmbd, False)
        ngram_delta = [0.4, 0.8, 0.9, 1.0]
        eval_ngram(ngram_delta, True)

    if '-eval_net' in sys.argv:
        results = eval_net()
        with open(path() + ".json", 'w+') as json_file:
            json.dump(results, json_file, indent=2)
        plot_log_loss(results["epoch_validation_logloss"],
                      results["epoch_train_logloss"], path() + ".pdf")


if __name__ == '__main__':
    main()
