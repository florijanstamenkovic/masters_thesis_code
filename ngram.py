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
from scipy.sparse import coo_matrix
from time import time

log = logging.getLogger(__name__)

_NGRAM_DIR = 'ngram_models'


class NgramModel():

    #   the directory where ngrams are stored
    @staticmethod
    def dir(tree, ftr_use, ts_reduction, min_occ, min_files):
        dir = "%s_features-%s_data-subset_%r-min_occ_%r-min_files_%r" % (
            "tree" if tree else "linear",
            "".join([str(int(b)) for b in ftr_use]),
            ts_reduction, min_occ, min_files)
        dir = os.path.join(_NGRAM_DIR, dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        return dir

    """
    An ngram based language model. Supports linear
    and dependancy-syntax-based ngrams.
    """

    @staticmethod
    def get(n, use_tree, feature_use, feature_sizes, trainset, dir, lmbd=0.1):
        """
        Gets an ngram model for the given parameters. First attempts
        to load a chached version of the model, if unable to load it,
        it trains a model and stores it for later usage.
        """

        #   try loading the model
        path = os.path.join(dir, "%d-grams_%.2f-smoothing.pkl" % (n, lmbd))
        model = util.try_pickle_load(path)
        if model is not None:
            return model

        #   failed to load model, create and train it
        train_start_time = time()
        model = NgramModel(n, use_tree, feature_use, feature_sizes, lmbd)
        model.train(trainset)
        log.info("Model trained in %.2f sec", time() - train_start_time)

        #   store model and then return it
        util.try_pickle_dump(model, path)
        return model

    def reduced_ngrams_mul(self):
        """
        Calculates how an n-dimensional matrix of ngrams can be
        falttened into a 2-dimensional matrix, so that sparse
        matrix operations can be done.

        :return: A tuple of form (shape, mapping). The 'shape'
            part is just a tuple defining the shape of the sparse
            matrix to be used when working with 2d-n-grams. The
            'mapping' part is a mapping matrix of shape [2, n]
            that can be used (dot product) to map ngrams from their
            original n-dimensional space to a 2D space.
        """

        #   cache this because it is used often
        if hasattr(self, '_reduced_ngrams_val'):
            return self._reduced_ngrams_val

        log.debug("ngram reduction calc, n=%d, sizes=%r",
                  self.n, self.feature_sizes)

        #   the shape of the n-gram array
        dims = [t[0] for t in zip(self.feature_sizes, self.feature_use)
                if t[1]]
        dims = np.tile(np.array(dims, dtype='uint32'), self.n)

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

    def __init__(self, n, tree, feature_use, feature_sizes, lmbd):
        """
        Initializes the ngram model. Does not train it.

        :param n: Number of tokens (words) that constitute an n-gram.
        :param tree: If n-grams should be generated from the dependency
            syntax tree. If False n-grams are generated in a linear way
            (the common definitinon of word n-grams).
        :param feature_use: An array of booleans which indicate which of the
            features should be used in this model.
        :param feature_sizes: An array of ints that indicate the dimension
            sizes of all the features.
        :param lmbd: Lambda parameter for additive (Laplace) smoothing.
        """

        self.n = n
        self.tree = tree
        self.feature_use = feature_use
        self.feature_sizes = feature_sizes
        self.lmbd = lmbd

    def train(self, ngrams):
        """
        Trains the model on the given data. Training boils down to
        counting ngram occurences, which are then stored in
        'self.counts'.

        :param trainset: An iterable of texts. Each text
            is a tokens tuple (feature_1, feature_2, ..., parent_ind)
            as returned by the 'data.process_string' function.
        """

        log.info("Training %d-gram model", self.n)

        #   calculate the shape of the accumulator
        #   and convert ngrams from (N, n) shape to (N, 2)
        cnt_shape, multipliers = self.reduced_ngrams_mul()
        ngrams = np.dot(ngrams, multipliers.T)

        #   create the accumulator
        counts = coo_matrix(
            (np.ones(ngrams.shape[0], dtype='uint8'),
                (ngrams[:, 0], ngrams[:, 1])), shape=cnt_shape,
            dtype='uint32').tocsc()

        log.info("Counting done, summing up")
        self.counts = counts
        self.count_sum = counts.sum()
        self.prob_normalizer = self.count_sum + \
            self.lmbd * np.prod(map(float, counts.shape))

    def probability(self, ngrams):
        """
        Calculates and returns the probability of
        a series of ngrams.

        :param tokens: A standard tuple of tokens:
            (feature_1, feature_2, ... , parent_inds)
            as returned by 'data.process_string' function.
        :return: numpy array of shape (ngram_count, 1)
        """
        _, multipliers = self.reduced_ngrams_mul()
        ngrams = np.dot(ngrams, multipliers.T)

        probs = map(lambda ind: self.counts[tuple(ind)], ngrams)
        #   depending on n, counts can be returned as arrays or floats
        #   so take care to convert to float
        probs = [(p if isinstance(p, float) else p.sum()) + self.lmbd
                 for p in probs]
        return np.array(probs) / self.prob_normalizer

    def __str__(self):
        return "%d-grams_%.2f-smoothing" % (self.n, self.lmbd)


class NgramAvgModel():

    """
    A model that averages n, (n-1), ... , 1 gram probabilities
    of a given sequences. Averaging is weighted.
    """

    @staticmethod
    def get(n, use_tree, feature_use, feature_sizes, trainset, dir, lmbd=0.1,
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

        models = [NgramModel.get(
            n, use_tree, feature_use, feature_sizes,
            trainset, dir, lmbd) for x in xrange(n, 0, -1)]

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

    def __str__(self):
        r_val = "ngram-averaging model, made of:"
        for w, m in zip(self.weights, self.models):
            r_val += "\n\t%.2f * %s" % (w, m)
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
        -t : Use tree-grams.
        -u FTRS : Features to use. FTRS must be a string composed of zeros
            and ones, of length 5. Ones indicate usage of following features:
            (word, lemma, google_pos, penn_pos, dependency_type), respectively.
    """

    logging.basicConfig(level=logging.INFO)
    log.info("Language modeling task - baselines")

    #   get the data handling parameters
    ts_reduction = util.argv('-s', None, int)
    min_occ = util.argv('-o', 1, int)
    min_files = util.argv('-f', 1, int)
    #   features to use, by default use only vocab
    #   choices are: [vocab, lemma, lemma-4, pos-google, pos-penn, dep-type]
    ft_format = lambda s: map(
        lambda s: s.lower() in ["1", "true", "yes", "t", "y"], s)
    ftr_use = np.array(util.argv('-u', ft_format("100000"), ft_format))
    tree = '-t' in sys.argv

    #   the directory where ngrams are stored
    dir = NgramModel.dir(tree, ftr_use, ts_reduction, min_occ, min_files)

    #   store logs
    if '-e' in sys.argv or '-es' in sys.argv:
        log_name = os.path.join(dir, "eval.log")
        log.addHandler(logging.FileHandler(log_name))

    #   create different n-gram models with plain +1 smoothing
    params = [
        (NgramModel, 1),
        (NgramModel, 2),
        (NgramModel, 3),
        (NgramModel, 4)
    ]

    #   create averaging n-gram models
    if '-a' in sys.argv:
        params.extend([
            (NgramAvgModel, 3),
            (NgramAvgModel, 4),
        ])

    for model_type, n in params:

        #   load the ngram data, answers, feature sizes etc
        sent_ngrams, qg_ngrams, answers, ftr_sizes = data.load_ngrams(
            n, ftr_use, tree, ts_reduction, min_occ, min_files)

        #   get the model
        model = model_type.get(
            n, tree, ftr_use, ftr_sizes, sent_ngrams, dir)

        if '-e' in sys.argv or '-es' in sys.argv:

            #   evaluation helper functions
            answ = lambda q_g: np.argmax([model.probability(q) for q in q_g])
            score = lambda a, b: (a == b).sum() / float(len(a))

            #   indices of questions used in evaluation
            #   different if we are checking a subset (-es flag)
            if '-es' in sys.argv:
                q_inds = np.arange(util.argv('-es', 50, int))
            else:
                q_inds = np.arange(answers.size)

            #   evaluate model
            log.info("Evaluating model: %s", model)
            eval_start_time = time()
            answers2 = [answ(qg_ngrams[i]) for i in q_inds]
            log.info("\tScore: %.4f", score(answers[q_inds], answers2))
            log.info("\tEvaluation time: %.2f sec", time() - eval_start_time)


if __name__ == "__main__":
    main()
