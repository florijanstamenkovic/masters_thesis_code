"""
Module for evaluating ngram models on the Microsoft
Sentences Completion Challenge dataset (obtained through
the 'data' module).
"""

import logging
import data
import sys
import itertools
import os
from time import time
from weakref import WeakValueDictionary

import numpy as np
from scipy.sparse import coo_matrix

import util

log = logging.getLogger(__name__)


class Counts(object):

    """
    Encapsulation for the counter of ngrams.
    """

    #   root of the directory where ngram counts are stored
    _COUNTS_DIR = 'counts_ngram'

    #   weak reference cache of loaded models
    _COUNTS_CACHE = WeakValueDictionary()

    @classmethod
    def get(cls, n, use_tree, feature_use, feature_sizes,
            ts_reduction, min_occ, min_files, trainset):
        """
        Gets an ngram model for the given parameters. First attempts
        to load a chached version of the model, if unable to load it,
        it trains a model and stores it for later usage.
        """

        #   directory where the counts are stored
        dir = "%s_features-%s_data-subset_%r-min_occ_%r-min_files_%r" % (
            "tree" if use_tree else "linear",
            "".join([str(int(b)) for b in feature_use]),
            ts_reduction, min_occ, min_files)
        dir = os.path.join(cls._COUNTS_DIR, dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        #   path where the model is stored
        path = os.path.join(dir, "%d-grams.pkl" % n)

        #   see if it's already loaded
        if path in cls._COUNTS_CACHE:
            return cls._COUNTS_CACHE[path]

        #   model not loaded in memory, try to load it from permanent storage
        model = util.try_pickle_load(path)
        if model is not None:
            return model

        #   failed to load counts, create them
        counts = Counts(np.array(feature_sizes)[feature_use], trainset)

        #   store counts into caches and then return it
        util.try_pickle_dump(counts, path)
        cls._COUNTS_CACHE[path] = counts
        return counts

    def __init__(self, feature_sizes, ngrams):
        """
        Creates the ngram counter. n is infered from
        'feature_sizes' and 'ngrams'.

        :param feature_sizes: An array of ints that indicate the dimension
            sizes of all the features in the ngrams.
        :param ngrams: The ngrams to count. A numpy array of shape
            (N, feature_count * n).
        """
        super(Counts, self).__init__()

        self.n = ngrams.shape[1] / feature_sizes.size
        self.feature_sizes = feature_sizes

        log.info("Counting %d-grams", self.n)

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
        self.count_sum = ngrams.shape[0]

    def count(self, ngrams):
        """
        Returns the counts of given ngrams.

        :param ngrams: A numpy array of ngrams of shape (N, n * feature_count).
        :return: A numpy array of counts for given ngrams, of shape (N, ).
        """
        ngrams = np.dot(ngrams, self.reduced_ngrams_mul()[1].T)
        counts = map(lambda ngram: self.counts[tuple(ngram)], ngrams)
        return np.array(counts, dtype='uint32')

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
        dims = np.tile(np.array(self.feature_sizes, dtype='uint32'), self.n)

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


class NgramModel():

    """
    An ngram based language model. Supports linear
    and dependancy-syntax-based ngrams.
    """

    def __init__(self, n, use_tree, feature_use, feature_sizes,
                 ts_reduction, min_occ, min_files, lmbd, trainset):
        """
        Initializes the ngram model. Does not train it.

        :param n: Number of tokens (words) that constitute an n-gram.
        :param tree: If n-grams should be generated from the dependency
            syntax tree. If False n-grams are generated in a linear way
            (the common definitinon of word n-grams).
        :param feature_sizes: An array of ints that indicate the dimension
            sizes of features used by this model.
        :param lmbd: Lambda parameter for additive (Laplace) smoothing.
        """

        #   count ngram occurences
        self.counts_n = Counts.get(
            n, use_tree, feature_use, feature_sizes,
            ts_reduction, min_occ, min_files, trainset)

        #   remember the number of ngrams in trainset
        self.counts_sum = trainset.shape[0]

        #   for all but unigrams, also count the occurences
        #   of the conditioning term
        if n > 1:
            self.counts_nmin1 = Counts.get(
                n - 1, use_tree, feature_use, feature_sizes,
                ts_reduction, min_occ, min_files, trainset[:, 1:])

        #   remember model hyperparameters
        self.n = n
        self.feature_sizes = np.array(feature_sizes)[feature_use]
        self.lmbd = lmbd

    def probability(self, ngrams):
        """
        Calculates and returns the probability of
        a series of ngrams.

        :param tokens: A standard tuple of tokens:
            (feature_1, feature_2, ... , parent_inds)
            as returned by 'data.process_string' function.
        :return: numpy array of shape (ngram_count, 1)
        """
        c_n = self.counts_n.count(ngrams)
        if self.n == 1:
            c_nmin1 = self.counts_sum
        else:
            c_nmin1 = self.counts_nmin1.count(ngrams[:, 1:])

        normalizer = self.lmbd * np.prod(self.feature_sizes)
        return (c_n + self.lmbd) / (c_nmin1 + normalizer)

    def __str__(self):
        return "%d-grams_%.2f-smoothing" % (self.n, self.lmbd)


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
    feature_format = lambda s: map(
        lambda s: s.lower() in ["1", "true", "yes", "t", "y"], s)
    feature_use = np.array(
        util.argv('-u', feature_format("100000"), feature_format))
    use_tree = '-t' in sys.argv

    #   store logs
    if '-e' in sys.argv or '-es' in sys.argv:
        log.addHandler(logging.FileHandler("ngram_eval.log"))

    #   create averaging n-gram models
    if '-a' in sys.argv:
        raise "Should implement this"

    for n in xrange(4, 0, -1):

        #   load the ngram data, answers, feature sizes etc
        sent_ngrams, qg_ngrams, answers, feature_sizes = data.load_ngrams(
            n, feature_use, use_tree, ts_reduction, min_occ, min_files)

        #   get the model
        model = NgramModel(n, use_tree, feature_use, feature_sizes,
                           ts_reduction, min_occ, min_files, 0.1, sent_ngrams)

        if '-e' in sys.argv or '-es' in sys.argv:

            #   evaluation helper functions
            answ = lambda q_g: np.argmax(
                [model.probability(q).prod() for q in q_g])
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
