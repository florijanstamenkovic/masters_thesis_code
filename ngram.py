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
    def params_to_string(n, tree_ngrams, feature_use, lmbd):
        """
        Returns a textual description of the ngram model that
        identifies it uniquely with respect to model hyperparameters.
        Can be used for file naming when storing models.
        """
        return "%d-grams_%s_features-%s_%.2f-smoothing" % (
            n, "tree" if tree_ngrams else "linear",
            "".join([str(int(b)) for b in feature_use]),
            lmbd)

    @staticmethod
    def get(n, lmbd, feature_use, features, parent_inds, path=None):
        """
        Gets an ngram model for the given parameters. First attempts
        to load a chached version of the model, if unable to load it,
        it trains a model and stores it for later usage.
        """

        #   the directory where we store cached models
        #   create it if necessary
        if path is None:
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

        #   a list of (mask, (shape)), tuples
        res = map(lambda c: (c, (prod(dims[c]),
                                 prod(dims[np.logical_not(c)]))), combs)
        #   filter out the invalid ones (first dim too large)
        res = filter(lambda r: r[1][0] < sys.maxint, res)
        #   the best mask has a large second dimension (for sparse-csc speed),
        #   but not too large (for storage optimization)
        best = res[np.argmin(map(lambda t: abs(t[1][1] - 1e8), res))][0]

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
                             self.dep_type_size is not None, *tokens,
                             invalid_tokens=[self.vocab_size - 1])

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
        log.info("Creating accumulator of shape %r, counting occurences",
                 counts.shape)

        #   go through the training files
        for ind, train_file in enumerate(train_texts):
            log.debug("Counting occurences in train text #%d", ind)
            ngrams = self.tokens_to_ngrams(train_file)
            log.debug("%d-grams shape: %d", self.n, ngrams.shape[0])
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


class NgramAvgModel():

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

        return NgramAvgModel(models, weights)

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

    Allowed cmd-line flags:
        -a : Also use averaging ngram models.
        -e : Do model evaluation (as opposed to just counting them)
        -s TS_FILES : Uses the reduced trainsed (TS_FILES trainset files)
        -o MIN_OCCUR : Only uses terms that occur MIN_OCCUR or more times
            in the trainset. Other terms are replaced with a special token.
        -f MIN_FILES : Only uses terms that occur in MIN_FILES or more files
            in the trainset. Other terms are replaced with a special token.
    """

    logging.basicConfig(level=logging.INFO)
    log.info("Language modeling task - baselines")

    #   get the data handling parameters
    ts_reduction = util.argv('-s', None, int)
    min_occur = util.argv('-o', 1, int)
    min_files = util.argv('-f', 1, int)
    bool_format = lambda s: s.lower() in ["1", "true", "yes", "t", "y"]
    tree_ngrams = util.argv('-t', True, bool_format)
    #   features to use, by default use only vocab
    #   choices are: [vocab, lemma, pos-google, pos-penn, dep-type]
    ft_format = lambda s: map(bool_format, s)
    features_use = np.array(util.argv('-u', ft_format("10000")), ft_format)

    log.info("Loading data")
    trainset, question_groups, answers = data.load_spacy(
        ts_reduction, min_occur, min_files)

    #   get features and parent inds from total trainset info
    features = [tf[:-1] for tf in trainset]
    parent_inds = [tf[-1] for tf in trainset]
    feature_sizes = [[a.max() for a in ft[:-1]] for ft in features]
    feature_sizes = feature_sizes.max(axis=0) + 1
    log.info("Feature sizes: %r", feature_sizes)

    #   folder where the ngram models are cached
    #   it is specific to data handling parameters
    path = os.path.join(
        'ngram_models', 'subset_%d-min_occ_%d-min_files_%d' % (
            ts_reduction, min_occur, min_files))
    if not os.path.exists(path):
        os.makedirs(path)

    #   log the loading process also to a file
    log_name = os.path.join(path, "info.log")
    log.addHandler(logging.FileHandler(log_name))

    #   create different n-gram models with plain +1 smoothing
    params = [
        (NgramModel, 1, False, v_size, dt_size, 1.0, trainset, path),
        (NgramModel, 2, False, v_size, dt_size, 1.0, trainset, path),
        (NgramModel, 3, False, v_size, dt_size, 1.0, trainset, path),
        (NgramModel, 4, False, v_size, dt_size, 1.0, trainset, path),
        (NgramModel, 2, True, v_size, None, 1.0, trainset, path),
        (NgramModel, 3, True, v_size, None, 1.0, trainset, path),
        (NgramModel, 4, True, v_size, None, 1.0, trainset, path),
        (NgramModel, 2, True, v_size, dt_size, 1.0, trainset, path),
        (NgramModel, 3, True, v_size, dt_size, 1.0, trainset, path),
        (NgramModel, 4, True, v_size, dt_size, 1.0, trainset, path),
    ]

    #   create averaging n-gram models
    if '-a' in sys.argv:
        params.extend([
            (NgramAvgModel, 3, False, v_size, dt_size, 1.0, trainset, path),
            (NgramAvgModel, 4, False, v_size, dt_size, 1.0, trainset, path),
            (NgramAvgModel, 3, True, v_size, None, 1.0, trainset, path),
            (NgramAvgModel, 4, True, v_size, None, 1.0, trainset, path),
            (NgramAvgModel, 3, True, v_size, dt_size, 1.0, trainset, path),
            (NgramAvgModel, 4, True, v_size, dt_size, 1.0, trainset, path)
        ])

    #   create the models
    for p in params:
        p[0].get(*p[1:])

    #   evaluation of ngram models
    if '-e' in sys.argv:
        log.info("Evaluating ngram models")

        #   helper function for evaluation
        score = lambda a, b: (a == b).sum() / float(len(a))

        #   function for getting the answer index (max-a-posteriori)
        answ = lambda q_g: np.argmax([model.probability(q) for q in q_g])
        for p in params:
            model = p[0].get(*p[1:])
            answers2 = [answ(q_g) for q_g in question_groups]
            log.info("Model: %s, score: %.4f", model.description(),
                     score(answers, answers2))


if __name__ == "__main__":
    main()
