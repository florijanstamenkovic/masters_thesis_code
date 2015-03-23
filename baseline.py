import logging
import data
import numpy as np
from scipy.sparse import dok_matrix
import multiprocessing
from multiprocessing import Process, Queue

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    log.info("Language modeling task - baselines")

    log.info("Loading data")
    train_files, question_groups, answers = data.load_spacy()
    vocab_size = max([tf[0].max() for tf in train_files]) + 1
    dep_type_size = max([tf[2].max() for tf in train_files]) + 1
    log.info("Vocabulary size: %d, dependancy type size: %d",
             vocab_size, dep_type_size)

    #   helper function for evaluation
    score = lambda a, b: (a == b).sum() / float(len(a))

    #   helper function for n-gram eval
    def ngram_eval(n, use_tree, use_dep_type):
        assert n > 1 and n < 5, "Bigrams to 4-grams allowed"

        log.info("\nBasline: %d-grams, use_tree: %d, use_dep_type: %d",
                 n, use_tree, use_dep_type)
        log.info("Counting bigram occurences")

        #   calculate the dimensions of the accumulator
        dim1 = vocab_size ** 2
        dim2 = vocab_size ** (n - 2)
        if use_tree and use_dep_type:
            dim1 *= dep_type_size ** ((n + 1) / 2)
            dim2 *= dep_type_size if n == 4 else 1

        #   create the accumulator
        counts = dok_matrix((dim1, dim2), dtype='uint32')
        log.info("Creating accumulator of shape %r", counts.shape)

        #   function for translating token indices to n-grams
        def tokens_to_ngrams(tokens):
            ngrams = data.ngrams(n, use_tree, use_dep_type, *tokens)
            ngrams = data.reduce_ngrams(
                ngrams, vocab_size,
                dep_type_size if use_tree and use_dep_type else None)
            return ngrams

        #   go through the training files
        for ind, train_file in enumerate(train_files):
            log.info("Counting occurences in train file #%d", ind)
            ngrams = tokens_to_ngrams(train_file)
            log.info("Number of %d-grams: %d", n, ngrams.shape[0])

            for ind in np.nditer(tuple(ngrams.T)):
                counts[ind] += 1

        #   the total number of ngrams with +1 smoothing
        log.info("Calculating sum")
        count_sum = sum(counts.itervalues()) + np.prod(counts.shape)
        log.info("Sum is: %d", count_sum)

        log.info("Calculating answer probabilities")
        prob = lambda s: np.prod(
            (counts[tuple(tokens_to_ngrams(s).T)] + 1) / count_sum)
        answ = lambda q_group: np.argmax([prob(q) for q in q_group])
        answers2 = [answ(q_group) for q_group in question_groups]
        log.info("Bigram probability score: %.4f", score(answers, answers2))

    # ngram_eval(1, False, False)
    # ngram_eval(2, False, False)
    # ngram_eval(3, False, False)
    # ngram_eval(2, True, False)
    # ngram_eval(3, True, False)
    ngram_eval(2, True, True)


if __name__ == "__main__":
    main()
