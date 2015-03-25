"""
Module for evaluating ngram models on the Microsoft
Sentences Completion Challenge dataset (obtained through
the 'data' module).
"""

import logging
import data
import numpy as np
import sys
import os
import util
from scipy.sparse import dok_matrix
# import multiprocessing
# from multiprocessing import Process, Queue

log = logging.getLogger(__name__)


class NgramModel():

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

    """
    An ngram based language model. Supports linear
    and dependancy-syntax-based ngrams.
    """

    def reduced_ngrams_mul(self):
        """
        Calculates how an n-gram array can be reduced in
        dimensionality. This is useful when using scipy-s
        sparse arrays because they only support 2D arrays.

        Calculation is based on vocab_size (because it determines
        the maximum values for vocab indices). If dependancy
        types are used in the ngrams, then the 'dep_type_size'
        param should be set to the appropriate value, otherwise
        it is assumed that all n-gram columns are vocab indices.

        Returns a tuple (dim_groups, muls). The 'dim_groups'
        object is a list of same length as the minimum possible
        number of ngram dimensions. Each element in list is
        another list, containing the maximum values for original
        columns (see example). The 'muls' indicates how the
        original colums should be multiplied when combining
        into a reduced column. It is a lost of the same length
        as the minimum possible number of ngram dimensions. Each
        element in list is another list, contaning the factors of
        multiplication for the appropriate original column.

        Example: Assuming that the maximum allowed index is
        2000, and for parameters (n=3, vocab_size=10, dep_type_size=3),
        the result will be:

        ([[20, 3, 20], [3, 20]], [[60, 3, 1], [20, 1]])

        Notice that 20 * 3 * 20 = 1200 < 2000, but the next column
        of max=3 whould result in 20*3*20*3=3600 > 2000.
        """

        log.debug("ngram reduction calc, n=%d, vocab=%d, dep_type=%r",
                  self.n, self.vocab_size, self.dep_type_size)

        #   calc the shape of the n-gram array
        n_shape = []
        for i in xrange(self.n):
            if i > 0 and self.use_tree and self.dep_type_size is not None:
                n_shape.append(self.dep_type_size)
            n_shape.append(self.vocab_size)

        #   group shape dimensions so that they fit
        dim_groups = [[]]
        dim_group = dim_groups[0]
        for dim in n_shape:
            if dim * np.prod([float(x) for x in dim_group]) >= sys.maxint:
                dim_group = []
                dim_groups.append(dim_group)
            dim_group.append(dim)

        #   calculate multipliers for each grouped column
        def mul_from_dim(dim):
            mul = [np.prod(dim[i + 1:]) for i in xrange(len(dim) - 1)]
            mul.append(1)
            return mul
        muls = map(mul_from_dim, dim_groups)

        log.debug("ngram reduction calc result: (%r, %r)", dim_groups, muls)
        return dim_groups, muls

    def reduce_ngrams(self, ngrams):
        """
        A function for reducing the dimensionality of an ngram
        array. Useful for representing 4-grams or 3-grams as 2-grams
        due to scipy's limitation of 2D-only sparse arrays.
        Utlizes the 'reduced_ngrams_mul' function for determening how
        to combine columns in the reduction.

        If dep_type_size param is None, it is assumed that dependency
        types are not used, and that all columns in ngrams are vocabulary
        indices. If it is not None, it is assumed that vocabulary indices
        and dependency types are interleaved, starting with vocabulary
        indices.

        :param ngrams: A numpy array of ngrams. Shape is (N, dims) where
            N is the number of ngrams (samples), and dims is the number of
            dimensions used (for example, 3-grams without dependency types
            have dims=3, but when with dependency types dims=5).
        :return: An array of reduced dimensionality that is a bijected
            translation of the original ngrams array.
        """

        #   calculate how the columns will be organized in the reduced array
        dim_groups, multipliers = self.reduced_ngrams_mul()

        #   prepare the return value array
        r_val = np.zeros((ngrams.shape[0], len(multipliers)), dtype=int)

        #   iterate through the original columns
        #   and add it into the new array, multiplied appropriately
        cols_done = 0
        for ind, muls in enumerate(multipliers):
            for mul in muls:
                r_val[:, ind] += mul * ngrams[:, cols_done]
                cols_done += 1

        return r_val

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
        return self.reduce_ngrams(ngrams)

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

        #   calculate the dimensions of the accumulator
        dim_groups, _ = self.reduced_ngrams_mul()

        #   create the accumulator
        count_shape = tuple([np.prod(g) for g in dim_groups])
        if len(count_shape) == 1:
            count_shape += (1,)
        counts = dok_matrix(count_shape, dtype='uint32')
        log.info("Creating accumulator of shape %r", counts.shape)
        count_sum = 0

        #   go through the training files
        for ind, train_file in enumerate(train_texts):
            log.info("Counting occurences in train text #%d", ind)
            ngrams = self.tokens_to_ngrams(train_file)
            log.info("Number of %d-grams: %d", self.n, ngrams.shape[0])

            for ind in np.nditer(tuple(ngrams.T)):
                counts[ind] += 1
            count_sum += ngrams.shape[0]

        self.counts = counts
        self.count_sum = float(sum(counts.itervalues()))
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

    def __getstate__(self):
        """
        Overriding pickling because scipy.sparse.DOK does not
        unpickle well, so we convert to COO. We need to keep
        the DOK in our objects because it supports more indexing
        options then COO. Scipy.sparse seems a bit crappy.
        """
        state = dict()
        state.update(self.__dict__)
        state["counts"] = self.counts.tocoo()
        return state

    def __setstate__(self, state):
        """
        Overriding pickling because scipy.sparse.DOK does not
        unpickle well, so we convert to COO. We need to keep
        the DOK in our objects because it supports more indexing
        options then COO. Scipy.sparse seems a bit crappy.
        """
        self.__dict__.update(state)
        self.counts = self.counts.todok()


class NgramAveragingModel():

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
        # NgramModel.get(1, False, voc_len, dep_t_len, 1.0, trainset),
        # NgramModel.get(2, False, voc_len, dep_t_len, 1.0, trainset),
        # NgramModel.get(3, False, voc_len, dep_t_len, 1.0, trainset),
        # NgramModel.get(4, False, voc_len, dep_t_len, 1.0, trainset),
        # NgramModel.get(2, True, voc_len, None, 1.0, trainset),
        # NgramModel.get(3, True, voc_len, None, 1.0, trainset),
        # NgramModel.get(4, True, voc_len, None, 1.0, trainset),
        # NgramModel.get(2, True, voc_len, dep_t_len, 1.0, trainset),
        # NgramModel.get(3, True, voc_len, dep_t_len, 1.0, trainset),
        # NgramModel.get(4, True, voc_len, dep_t_len, 1.0, trainset),
    ]

    #   create averaging n-gram models
    models.extend([
        NgramAveragingModel.get(3, False, voc_len, dep_t_len, 1.0, trainset),
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
