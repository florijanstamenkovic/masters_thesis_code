"""
Module for loading and processing the raw data of the
Microsoft Sentence Completion Challenge (stored in the ./data/ folder).
Utilizes the spaCy library for tokenizing and parsing text.
"""

import logging
import os
import numpy as np
import util
import re
import codecs
import spacy.en


log = logging.getLogger(__name__)

#   the tokenizer / parser used
nlp = spacy.en.English()


def preprocess_string(string, tolower=True):
    """
    Does string preprocessing. Typically this means converting
    multiple spaces to single spaces, digits to special number
    tokens etc. Returns the processed string.

    param string: The string to process.

    param tolower: If the string should be converted to lowercase.
    """

    #   convert all whitespace to single spaces
    string = re.sub(r"\s+", " ", string)

    #   convert all multiple dot occurences to elipses
    string = re.sub(r"\.\.+", "...", string)

    #   convert dashes and underscores to spaces
    string = re.sub(r"[\-\_]+", " ", string)

    #   remove periods some common abbreviations
    string = re.sub(r"[mM]r\.", "mr", string)
    string = re.sub(r"[mM]rs\.", "mrs", string)

    #   convert numbers to a number token
    string = re.sub(r"[\.\,\d]*\d", "NUMBER_TOK", string)

    if tolower:
        string = string.lower()

    return string


def process_string(string, process=True):
    """
    Helper function for processing a string.
    First it (optionally) pre-processes the string.
    Then it tokenizes and parses the file using spaCy.
    Finally it extracts vocabulary indices, dependency
    head indices and dependency type indices.
    All are numpy arrays.

    :param string: A string of text.
    :return: (vocab_indices, head_indices, dep_type_indices)
    """
    if process:
        string = preprocess_string(string)

    if isinstance(string, str):
        string = unicode(string)

    log.debug("Tokenizing and parsing string")
    tokens = nlp(string, True, True)
    misspelling = set([t.orth_ for t in tokens if t.prob == 0.0])
    log.debug("Misspelled words:\n\t%s", ", ".join(misspelling))

    log.debug("Converting tokens to numpy arrays")
    orth = np.array(tokens.to_array([spacy.en.attrs.ORTH]),
                    dtype='uint32').flatten()
    indices = dict(zip(tokens, xrange(len(tokens))))
    head = np.array([indices[t.head] for t in tokens],
                    dtype='uint32')
    dep_type = np.array([t.dep for t in tokens], dtype='uint8')
    return (orth, head, dep_type)


def load_spacy():
    """
    Loads the cached version of spaCy-processed data.
    Returns the same data as load_spacy_raw().
    """

    file_name = os.path.join("data", "processed_data.pkl")
    data = util.try_pickle_load(file_name)
    if data is not None:
        return data

    data = load_spacy_raw()

    util.try_pickle_dump(data, file_name)
    return data


def load_spacy_raw():
    """
    Loads the raw text data that constitutes the Microsoft
    Sentence Completion Challenge (stored in ./data/).
    Processes the data, tokenizes and parses it, and returns
    the results.

    Returned is a tuple (train_files, question_groups, answers).
    The 'train_files' object is a list of parsed-text-tuples,
    which will be described below. The 'question_groups' object
    is an iterable question groups. Each group consists of 5
    sentences (one of which is correct). Each sentence is a
    parsed-text-tuple. The 'answers'
    object is a numpy array of shape (question_group_count, )
    that contains the indices of the correct sentences in
    question groups. Finaly, the parsed-text-tuple is a tuple
    as returned by the process_string() function. It is of
    form (vocab_indices, head_indices, dep_type_indices).
    """

    #   process questions
    log.info("Processing questions file")
    with open(os.path.join("data", "questions.txt")) as q_file:
        questions = [l.strip() for l in q_file.readlines()]

    #   remove question numbering and brackets from questions
    questions = map(lambda s: s[s.find(" ") + 1:], questions)
    questions = map(lambda s: re.sub(r"[\[\]]", "", s), questions)
    map(log.debug, questions)

    #   preprocess, tokenize and parse questions
    questions = map(process_string, questions)

    #   group sentences of a single question into a single array
    question_groups = [questions[i * 5: (i + 1) * 5] for
                       i in xrange(len(questions) / 5)]

    #   process answers
    log.info("Processing answers file")
    with open(os.path.join("data", "answers.txt")) as q_file:
        answers = [l.strip() for l in q_file.readlines()]
    #   translate answer choice (i.e "42b)") into indices
    answers = map(lambda l: "abcde".find(l[l.find(" ") - 2]), answers)
    answers = np.array(answers, dtype='uint8')

    def load_train_file(train_file):
        """
        Helper function for processing a training text file.
        Strips away the Guttenberg stuff, concatenates lines
        etc. Then tokenizes the data, processes it
        using the data_from_tokens function and returns
        the results.

        :param train_file: Path to training file.
        :return: (vocab_indices, head_indices, dep_type_indices)
        """

        log.info("Processing training file %r", train_file)

        #   read file
        path = os.path.join(train_dir, train_file)
        with codecs.open(path, "rU", "latin1") as f:
            log.debug("Loading file")
            data = f.read()

        #   eliminate Guttenberg metadata
        data = data[data.rfind("*END*") + 5:]
        data = data[:data.find("End of Project Gutenberg's")]
        log.info("Training file has %d chars", len(data))

        r_val = process_string(data)
        log.info("Training file has %d tokens", r_val[0].shape[0])
        return r_val

    log.info("Processing training data")
    train_dir = os.path.join("data", "trainset")
    train_files = map(load_train_file, os.listdir(train_dir))

    return train_files, question_groups, answers


def ngrams(n, tree_ngram, use_deps, token_ind, parent_ind, dep_type,):
    """
    Converts token, head and dependeny indices into n-grams.
    Can create dependeny-syntax-tree-based n-grams and linear
    n-grams.

    Tree-based n-grams are formed by following the tree
    heirarchy upwards from a node. Dependency type can also be
    inserted into the n-gram. For example, if par(x) denotes the
    parent of 'x', and type(x) denotes the type of dependancy between
    'x' and par(x), then a 3-gram for any given 'x' can be
    (x, type(x), par(x), type(par(x)), par(par(x))) or
    (x, par(x), par(par(x))) if dependency types are used or not,
    respectively.

    Linear n-grams are plain old sequence based n-grams, organized
    backwards, so that for a sentence "a b c", the resulting bigrams
    would be [(b, a), (c, b)]. This organization is consistent with
    syntax-tree ngrams in the sense that the conditioned term is
    the 0-th element in the ngram, followed by it's closest conditioning
    term, and so forth.

    :param token_ind: Token indices of a text.
    :param parent_ind: Indices of node parents in the dependancy syntax
        tree. If a node has no parent (is root of a sentence), then it's
        parent_ind is the same as it's own index.
    :param dep_type: Dependency type of a node towards it's parent.
    :param n: Desired n-gram length.
    :param tree_ngram: If or not n-grams should be based on the
        dependancy tree structure.
    :param use_deps: If or not syntactic dependancies should be
        included in the n-gram (only applicable if tree_ngram is True).

    :return: Returns a numpy array of shape (count, n_gram_depth),
        where count is the number of resulting n-grams (depends on n),
        and n_gram_depth is the number of n_gram parameters (depends on n
        and if tree dependencies are used). Note that the 0-th column
        in the array is the conditioned term, 1-st colum is it's closest
        conditioning term, and so forth.
    """

    assert token_ind.size == parent_ind.size, \
        "Must have the same number of token indices and head indices"
    assert token_ind.size == dep_type.size, \
        "Must have the same number of token indices and dependency types"

    #   calculate the shape of the resulting array
    if tree_ngram:
        shape = (token_ind.size, (n * 2 - 1) if use_deps else n)
    else:
        shape = (token_ind.size - n + 1, n)

    #   init array, and set first column to be n-gram[0]
    r_val = np.zeros(shape, dtype='uint32')
    if tree_ngram:
        r_val[:, 0] = token_ind
    else:
        r_val[:, -1] = token_ind[:token_ind.size - n + 1]

    #   append other n-gram info
    for ind in xrange(1, n):

        if tree_ngram:
            if use_deps:
                r_val[:, ind * 2 - 1] = dep_type
                r_val[:, ind * 2] = token_ind[parent_ind]
            else:
                r_val[:, ind] = token_ind[parent_ind]

            #   move on with the heirarchy
            dep_type = dep_type[parent_ind]
            parent_ind = parent_ind[parent_ind]

        else:
            r_val[:, -1 - ind] = token_ind[ind:token_ind.size - n + 1 + ind]

    return r_val


def main():
    """
    Performs dataset loading / caching and prints out some
    information about the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    train_files, questions, answers = load_spacy()
    log.info("Number of works in trainset: %d", len(train_files))
    log.info("Trainset token count: %d", sum([len(x[0]) for x in train_files]))
    log.info("Number of questions: %d", len(questions))

if __name__ == "__main__":
    main()
