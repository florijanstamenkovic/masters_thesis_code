import logging
import data
import numpy as np
from scipy.sparse import dok_matrix

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)
    log.info("Language modeling task - baselines")

    log.info("Loading data")
    word_to_ind, ind_to_word, train_files, questions, answers = data.load()
    vocab_size = len(ind_to_word)

    #   helper function for evaluation
    score = lambda a, b: (a == b).sum() / float(len(a))

    log.info("Baseline: random selection")
    answers2 = np.random.randint(0, 5, len(answers))
    log.info("Random selection score: %.4f", score(answers, answers2))

    # log.info("Basline: unigram probability")
    # log.info("Counting term occurences")
    # counts = np.zeros((vocab_size, ), dtype='uint32')
    # for train_file_ind, train_file in enumerate(train_files):
    #     log.debug("Processing training file #%d", train_file_ind)
    #     counts += np.histogram(train_file, vocab_size, (0, vocab_size))[0]
    # freq = counts / float(counts.sum())
    # log.info("Calculating answer probabilities")
    # prob = lambda sentence: np.prod([freq[w_ind] for w_ind in sentence])
    # answ = lambda question: np.argmax([prob(s) for s in question])
    # answers2 = [answ(question) for question in questions]
    # log.info("Unigram probability score: %.4f", score(answers, answers2))

    log.info("Basline: bigram probability")
    log.info("Counting bigram occurences")

    #   sparsely count bigram occurences
    counts = dok_matrix((vocab_size, vocab_size), dtype='uint32')
    for train_file_ind, train_file in enumerate(train_files[:2]):
        log.debug("Processing training file #%d", train_file_ind)
        for i1, i2 in np.nditer([train_file[:-1], train_file[1:]]):
            counts[i1, i2] += 1

    #   the total number of bigrams
    count_sum = float(sum([(f.size - 1) for f in train_files]))
    #   we are using +1 smoothing
    count_sum += vocab_size ** 2.

    log.info("Calculating answer probabilities")
    prob = lambda s: np.prod((counts[s[:-1], s[1:]].todense() + 1) / count_sum)
    answ = lambda question: np.argmax([prob(s) for s in question])
    answers2 = [answ(question) for question in questions]
    log.info("Bigram probability score: %.4f", score(answers, answers2))


if __name__ == "__main__":
    main()
