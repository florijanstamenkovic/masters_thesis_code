import logging
import os
import re
import numpy as np
import util


log = logging.getLogger(__name__)

INTER_KEEP = ",.:;!?\""
INTER_STRIP = "'+-*/()[]{}"


def process_line(line, word_to_ind, ind_to_word):
    """
    Processes a single line of text. This implies tokenizing
    and returning a list of token indices. Vocabulary is
    augmented with potentionally unseen words.

    :param line: A line of text.
    :param word_to_ind: A dictionary mapping words to unique numbers.
    :param ind_to_word: Reverse mapping of unique numbers to words.
    """

    #   split line on delimiters, keep some, ditch others
    words = re.split(r"[\s\+\-\*\/\(\)\[\]\{\}]+|([,.:;!?\"]+)", line)
    words = filter(lambda s: s is not None and len(s) > 0, words)
    words = [w.lower() for w in words]

    #   analyze words individually
    def tokens_for_word(word):
        """
        Helper function, for a given word it returns a list
        of tokens. Typically the list will only contain that
        single word, but some processsing might result in multiple
        tokens for the given input.
        """
        #   strip interpuction we don't keep
        word = word.strip(INTER_STRIP)

        if len(word) == 0:
            return []

        #   replace digits with a special token
        word = re.sub(r"\d", "<DIG>", word)

        #   handle elipses in the start
        dot_count = 0
        while len(word) > dot_count and word[dot_count] == '.':
            dot_count += 1
        if dot_count >= 2:
            return ["<ELIPSIS>"] + tokens_for_word(word[dot_count:])

        #   handle elipsis in the end
        dot_count = 0
        while len(word) > dot_count and word[-1 - dot_count] == '.':
            dot_count += 1
        if dot_count >= 2:
            return tokens_for_word(word[:-dot_count]) + ["<ELIPSIS>"]

        #   separate interpuction we do keep
        if word[0] in INTER_KEEP:
            return [word[0]] + tokens_for_word(word[1:])
        if word[-1] in INTER_KEEP:
            return tokens_for_word(word[:-1]) + [word[-1]]

        return [word]

    words = sum([tokens_for_word(w) for w in words], [])
    log.debug("Tokens: %r", words)

    #   translate words into indices
    words_ind = []
    for word in words:
        ind = word_to_ind.get(word)
        if ind is None:
            ind = len(word_to_ind)
            word_to_ind[word] = ind
            ind_to_word.append(word)
        words_ind.append(ind)

    #   return a list of indices
    return words_ind


def load():
    """
    Loads the dataset, attempting to read it from
    the cached location on the hard drive.

    Returns the same value as the load_raw() function.
    """

    file_name = "processed_data.pkl"
    file_name = os.path.join("data", file_name)

    cached = util.try_pickle_load(file_name)
    if cached is not None:
        return cached

    cached = load_raw()

    util.try_pickle_dump(cached, file_name)
    return cached


def load_raw():
    """
    Reads the dataset (training texts and questions/answers),
    processes them into tokens and returns the following tuple:
    (word_to_ind, ind_to_word, train_files, questions, answers).

    word_to_ind - A dictionary mapping words to unique integers.
    ind_to_word - A list for reverse mapping on integers to words.
    train_files - A list of numpy arrays of token indices. One
        array for each training text.
    questions - A list of questions. Each question is a list of
        five numpy arrays. Each numpy array is a sequence of token
        indices of a sentence. Only one of the sentences in the
        question (of them five) is correct.
    answers - A numpy array of indices of correct answers.
    """

    word_to_ind = {}
    ind_to_word = []

    #   process questions
    log.info("Processing questions file")
    with open(os.path.join("data", "questions.txt")) as q_file:
        lines = [l.strip() for l in q_file.readlines()]

    #   remove question numbering from lines
    remove_first = lambda s: s[s.find(" ") + 1:]
    lines = [remove_first(l) for l in lines]
    map(log.debug, lines)

    #   tokenize questions
    lines = [process_line(l, word_to_ind, ind_to_word) for l in lines]

    #   group sentences of a single question into a single numpy array
    questions_np = []
    for current_ind in xrange(len(lines)):
        if (current_ind % 5) == 0:
            question = []
            questions_np.append(question)
        question.append(np.array(lines[current_ind], dtype='uint32'))

    #   process answers
    log.info("Processing answers file")
    with open(os.path.join("data", "answers.txt")) as q_file:
        lines = [l.strip() for l in q_file.readlines()]
    answers_np = np.array(
        ["abcde".find(l[l.find(" ") - 2]) for l in lines], dtype='uint32')

    log.info("Processing training data")
    train_dir = os.path.join("data", "trainset")
    train_files_np = []
    for train_file in os.listdir(train_dir):
        log.info("Processing training file %r", train_file)

        #   read file
        with open(os.path.join(train_dir, train_file)) as f:
            lines = [l.strip() for l in f.readlines()]

        #   eliminate Guttenberg data
        first_line = 0
        for ind, line in enumerate(lines):
            if line.startswith('*END*'):
                first_line = ind + 1
                break
        lines = lines[first_line:-1]

        #   replace multiple newlines with a single newline
        lines = [l for i, l in enumerate(lines) if
                 (i == 0 or len(l) > 0 or len(lines[i - 1]) > 0)]
        #   replace empty lines (newlines) with a special token
        lines = [l if len(l) > 0 else "<PARAGRAPH>" for l in lines]

        #   translate lines to word indices, augmenting vocabulary
        lines_ind = [process_line(l, word_to_ind, ind_to_word) for l in lines]
        #   prune out None lines, rare but possible
        lines_ind = filter(lambda l: l is not None, lines_ind)

        #   turn all the words from file into numpy array
        token_count = sum([len(l) for l in lines])
        train_file_np = np.zeros((token_count, ), "uint32")
        current_ind = 0
        for line_ind in lines_ind:
            for ind in line_ind:
                train_file_np[current_ind] = ind
                current_ind += 1

        train_files_np.append(train_file_np)

    return word_to_ind, ind_to_word, train_files_np, questions_np, answers_np


def main():
    """
    Performs dataset loading / caching and prints out some
    information about the dataset.
    """

    logging.basicConfig(level=logging.INFO)

    word_to_ind, ind_to_word, train_files_np, questions_np, answers_np = load()
    log.info("Vocabulary length: %d", len(word_to_ind))
    log.info("Number of works in trainset: %d", len(train_files_np))
    log.info("Trainset word count: %d", sum([len(x) for x in train_files_np]))
    log.info("Number of questions: %d", len(questions_np))

if __name__ == "__main__":
    main()
