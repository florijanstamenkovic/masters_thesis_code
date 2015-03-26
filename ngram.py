"""
Module for evaluating ngram models on the Microsoft
Sentences Completion Challenge dataset (obtained through
the 'data' module).
"""

import logging
import data
import numpy as np
import sys
import itertools
import os
import util
from scipy.sparse import csc_matrix, coo_matrix
# import multiprocessing
# from multiprocessing import Process, Queue

log = logging.getLogger(__name__)


class NgramModel():
    """
    An ngram based language model. Supports linear
    and dependancy-syntax-based ngrams.
    """

    @staticmethod
    def params_to_string(n, use_tree, dep_type_size, lmbd):
        """
        Returns a textual description of the ngram model that
        identifies it uniquely with respect to model hyperparameters.
        Can be used for file naming when storing models.
        """

        if n == 1:
            return "1-grams_%.2f-smoothing" % lmbd

        return "%d-grams_%s_with%s-deps_%.2f-smoothing" % (
            n, "tree" if use_tree else "linear",
            "out" if (not use_tree) or dep_type_size is None else "",
            lmbd)

    @staticmethod
    def get(n, use_tree, vocab_size, dep_type_size, lmbd, train_data):
        """
        Gets an ngram model for the given parameters. First attempts
        to load a chached version of the model, if unable to load it,
        it trains a model and stores it for later usage.

        All parameters are simply passed to the NgramModel constructor,
        except for 'train_data' which is used with 'NgramModel.train()'
        function.
        """

        #   the directory where we store cached models
        #   create it if necessary
        path = 'ngram_models'
        if not os.path.exists(path):
            os.makedirs(path)

        #   model name and path
        name = NgramModel.params_to_string(n, use_tree, dep_type_size, lmbd)
        log.info("Getting model: %s", name)
        path = os.path.join(path, name + ".pkl")

        #   try loading the model
        model = util.try_pickle_load(path)
        if model is not None:
            return model

        #   failed to load model, create and train it
        model = NgramModel(n, use_tree, vocab_size, dep_type_size, lmbd)
        model.train(train_data)

        #   store model and then return it
        util.try_pickle_dump(model, path)
        return model

    def reduced_ngrams_mul(self):
        """
        Calculates how an n-dimensional matrix of ngrams can be
        falttened into a 2-dimensional matrix, so that sparse
        matrix operations can be done. For performance benefits
        it is desired that the second dimension is minimal, while
        the first can not exceed 'sys.maxint'.

        :return: A tuple of form (shape, mapping). The 'shape'
            part is just a tuple defining the shape of the sparse
            matrix to be used when working with 2d-n-grams. The
            'mapping' part is a mapping matrix of shape [2, n]
            that can be used (dot product) to map ngrams from their
            origina n-dimensional space to a 2D space.
        """

        #   cache this because it is used often
        if hasattr(self, '_reduced_ngrams_val'):
            return self._reduced_ngrams_val

        log.debug("ngram reduction calc, n=%d, vocab=%d, dep_type=%r",
                  self.n, self.vocab_size, self.dep_type_size)

        #   calc the shape of the n-gram array
        vs, dts, n = self.vocab_size, self.dep_type_size, self.n
        if self.use_tree and dts is not None:
            dims = np.array((vs,) + (dts, vs) * (n - 1))
        else:
            dims = np.array((vs,) + (vs, ) * (n - 1))

        #   all possible combinations of dimensions
        combs = itertools.product((False, True), repeat=len(dims))
        combs = map(np.array, combs)

        def prod(it):
            """
            A product implementation that does not suffer
            from overflow.
            """
            r_val = 1
            for i in it:
                r_val *= int(i)
            return r_val

        #   a list of (mask, first_dim_cardinality), tuples
        #   filter out the invalid ones (first dim too large)
        #   and among the rest find the best
        res = map(lambda c: (c, prod(dims[c])), combs)
        res = filter(lambda r: r[1] < sys.maxint, res)
        best = res[np.argmax(map(lambda t: t[1], res))][0]

        def mult(dims, mask):
            """
            Generates multiplication masks from the
            given 'dims' array (original dimensions) and
            'mask' (indicates which dimensions should be used).
            """
            r_val = np.array(dims)
            r_val[np.logical_not(mask)] = 1
            for i in xrange(len(dims)):
                r_val[i] = np.prod(r_val[i + 1:])
            r_val[np.logical_not(mask)] = 0
            return r_val

        muls = np.array([mult(dims, best), mult(dims, np.logical_not(best))])
        shape = (prod(dims[best]), prod(dims[np.logical_not(best)]))

        log.debug("ngram reduction calc shape: %r, and multiplies: %r",
                  shape, muls)
        self._reduced_ngrams_val = (shape, muls)
        return (shape, muls)

    def __init__(self, n, use_tree, vocab_size, dep_type_size=None, lmbd=0.0):
        """
        Initializes the ngram model. Does not train it.

        :param n: Number of tokens (words) that constitute an n-gram.
        :param use_tree: If n-grams should be generated from the dependency
            syntax tree. If False n-grams are generated in a linear way
            (the common definitinon of word n-grams).
        :param vocab_size: Size of the vocabulary used in n-grams.
        :param dep_type_size: Number of different dependency types. Only
            utilized if 'use_tree' param is True. If set to None, dependency
            type information will not be included in the n-grams, even when
            using tree-n-grams.
        :param lmbd: Lambda parameter for additive (Laplace) smoothing.
        """

        self.n = n
        self.use_tree = use_tree
        self.vocab_size = vocab_size
        self.dep_type_size = dep_type_size
        self.lmbd = lmbd

    def tokens_to_ngrams(self, tokens):
        """
        Converts a tokens tuple (vocab_inds, head_inds, dep_type_inds)
        into a numpy n-gram array using the 'data.ngrams' function.
        Then it reduces n-gram array dimensionality using the
        'NgramModel.reduced_ngrams_mul' function. Returns the resulting
        numpy array.
        """
        ngrams = data.ngrams(self.n, self.use_tree,
                             self.dep_type_size is not None, *tokens)

        #   reduce ngrams to two dimensions (for sparse matrices to handle)
        _, multipliers = self.reduced_ngrams_mul()
        return np.dot(ngrams, multipliers.T)

    def train(self, train_texts):
        """
        Trains the model on the given data. Training boils down to
        counting ngram occurences, which are then stored in
        'self.counts'. Also the 'self.prob_normalizer' variable
        is set, which is used to convert counts to probabilities
        (it also includes additive smoothing).

        :param train_texts: An iterable of texts. Each text
            is a tokens tuple (vocab_indices, head_indices, dep_type_indices)
            as returned by the 'data.process_string' function.
        """

        log.info("Training %d-gram model", self.n)

        #   calculate the shape of the accumulator
        cnt_shape, _ = self.reduced_ngrams_mul()

        #   create the accumulator
        counts = csc_matrix(cnt_shape, dtype='uint32')
        log.info("Creating accumulator of shape %r", counts.shape)

        #   go through the training files
        for ind, train_file in enumerate(train_texts):
            log.info("Counting occurences in train text #%d", ind)
            ngrams = self.tokens_to_ngrams(train_file)
            log.info("%d-grams shape: %d", self.n, ngrams.shape[0])
            assert ngrams.ndim == 2, "Only 2D n-gram matrix allowed"

            #   count ngrams
            data = (np.ones(ngrams.shape[0]))
            counts += coo_matrix(
                (data, (ngrams[:, 0], ngrams[:, 1])), shape=cnt_shape).tocsc()

        log.info("Counting done, summing up")
        self.counts = counts
        self.count_sum = counts.sum()
        self.prob_normalizer = self.count_sum + \
            self.lmbd * np.prod(map(float, counts.shape))

    def probability(self, tokens):
        """
        Calculates and returns the probability of
        a series of tokens.

        :param tokens: A standard tuple of tokens:
            (vocab_indices, head_indices, dep_type_indices)
            as returned by 'data.process_string' function.
        """

        ngrams = self.tokens_to_ngrams(tokens)
        probs = map(lambda ind: self.counts[tuple(ind)], ngrams)
        probs = [(e if isinstance(e, float) else e.sum()) + 1 for e in probs]
        return np.prod(probs) / self.prob_normalizer

    def description(self):
        """
        Returns the same string as 'NgramModel.params_to_string',
        just passes it this model's parameters.
        """
        return NgramModel.params_to_string(
            self.n, self.use_tree, self.dep_type_size, self.lmbd)


class NgramAveragingModel():
    """
    A model that averages n, (n-1), ... , 1 gram probabilities
    of a given sequences. Averaging is weighted.
    """

    @staticmethod
    def get(n, use_tree, vocab_size, dep_type_size, lmbd, train_data,
            weight=0.5):
        """
        Gets an averaging ngram model for the given parameters.
        Obtains the normal (non-averaging) ngram models using
        'NgramModel.get'.

        All params except for 'weight' are passed to 'NgramModel.get'.
        The 'weighs' param is used to calculate averaging weights.
        If of type 'float', then the weights are calcualted in
        so that for weights=0.6 and n=3 the weights are [0.6, 0.24, 0.16],
        for [3-grams, 2-grams, 1-grams] respectively. If 'weights' is
        not a 'float', it is expected to be an iterable of n floats.
        """

        models = [NgramModel.get(x, use_tree, vocab_size, dep_type_size, lmbd,
                                 train_data) for x in xrange(n, 0, -1)]

        if isinstance(weight, float):
            weights = [1.0]
            for _ in xrange(n - 1):
                weights[-1] *= weight
                weights.append(1 - sum(weights))
        else:
            weights = weight

        return NgramAveragingModel(models, weights)

    def __init__(self, models, weights):
        log.info("Ngram averaging, max n=%d, weights=%r", len(models), weights)

        self.models = models
        self.weights = weights

    def probability(self, tokens):
        """
        Calculates and returns the probability of
        a series of tokens.

        :param tokens: A standard tuple of tokens:
            (vocab_indices, head_indices, dep_type_indices)
            as returned by 'data.process_string' function.
        """
        total = 0.0
        for model, weight in zip(self.models, self.weights):
            total += model.probability(tokens) * weight

        return total

    def description(self):
        """
        A textual description of the model. Uniquely describes the model,
        but not suitable for file names due to newlines.
        """
        r_val = "ngram-averaging model, made of:"
        for w, m in zip(self.weights, self.models):
            r_val += "\n\t%.2f * %s" % (w, m.description())
        return r_val


def main():
    """
    Trains and evaluates a few different ngram models
    on the Microsoft Sentence Completion Challenge.
    """

    logging.basicConfig(level=logging.INFO)
    log.info("Language modeling task - baselines")

    log.info("Loading data")
    trainset, question_groups, answers = data.load_spacy()
    voc_len = max([tf[0].max() for tf in trainset]) + 1
    dep_t_len = max([tf[2].max() for tf in trainset]) + 1
    log.info("Vocabulary size: %d, dependancy type size: %d",
             voc_len, dep_t_len)

    #   helper function for evaluationP@
    score = lambda a, b: (a == b).sum() / float(len(a))

    #   create different n-gram models with plain +1 smoothing
    models = [
        NgramModel.get(1, False, voc_len, dep_t_len, 1.0, trainset),
        NgramModel.get(2, False, voc_len, dep_t_len, 1.0, trainset),
        NgramModel.get(3, False, voc_len, dep_t_len, 1.0, trainset),
        NgramModel.get(4, False, voc_len, dep_t_len, 1.0, trainset),
        NgramModel.get(2, True, voc_len, None, 1.0, trainset),
        NgramModel.get(3, True, voc_len, None, 1.0, trainset),
        NgramModel.get(4, True, voc_len, None, 1.0, trainset),
        NgramModel.get(2, True, voc_len, dep_t_len, 1.0, trainset),
        NgramModel.get(3, True, voc_len, dep_t_len, 1.0, trainset),
        NgramModel.get(4, True, voc_len, dep_t_len, 1.0, trainset),
    ]

    #   create averaging n-gram models
    models.extend([
        # NgramAveragingModel.get(3, False, voc_len, dep_t_len, 1.0, trainset),
        # NgramAveragingModel.get(4, False, voc_len, dep_t_len, 1.0, trainset),
        # NgramAveragingModel.get(3, True, voc_len, None, 1.0, trainset),
        # NgramAveragingModel.get(4, True, voc_len, None, 1.0, trainset),
        # NgramAveragingModel.get(3, True, voc_len, dep_t_len, 1.0, trainset),
        # NgramAveragingModel.get(4, True, voc_len, dep_t_len, 1.0, trainset)
    ])

    #   evaluation functions
    answ = lambda q_group: np.argmax([model.probability(q) for q in q_group])
    for model in models:
        answers2 = [answ(q_group) for q_group in question_groups]
        log.info("Model: %s, score: %.4f", model.description(),
                 score(answers, answers2))


if __name__ == "__main__":
    main()
