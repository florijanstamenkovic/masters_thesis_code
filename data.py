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
    Finally it extracts vocabulary indices (orth),
    lemma indices (lemm), part of speech indices (pos),
    detailed part of speech tag indices (tag),
    syntax to-parent-dependency type indices (dep) and
    dependency head indices (head).
    All are numpy arrays.

    :param string: A string of text.
    :return: (orth, lemm, pos, tag, dep_type, head)
    """
    if process:
        string = preprocess_string(string)

    if isinstance(string, str):
        string = unicode(string)

    tokens = nlp(string, True, True)
    misspelling = set([t.orth_ for t in tokens if t.prob == 0.0])
    if len(misspelling) > 0:
        log.debug("Misspelled words:\n\t%s", ", ".join(misspelling))

    orth = np.array(tokens.to_array([spacy.en.attrs.ORTH]),
                    dtype='uint32').flatten()
    lemm = np.array(tokens.to_array([spacy.en.attrs.LEMMA]),
                    dtype='uint32').flatten()
    pos = np.array(tokens.to_array([spacy.en.attrs.POS]),
                   dtype='uint8').flatten()
    tag = np.array(tokens.to_array([spacy.en.attrs.TAG]),
                   dtype='uint8').flatten()
    dep_type = np.array([t.dep for t in tokens], dtype='uint8')

    #   convert head references to indices
    indices = dict(zip(tokens, xrange(len(tokens))))
    head = np.array([indices[t.head] for t in tokens],
                    dtype='uint32')

    return (orth, lemm, pos, tag, dep_type, head)


def load_spacy(subset=None, min_occ=1, min_files=1):
    """
    Loads the cached version of spaCy-processed data.
    Accepts the same parameters and returns the same
    data as load_spacy_raw().
    """

    dir = os.path.join("data", "processed")
    if not os.path.exists(dir):
            os.makedirs(dir)
    name_base = "subset_%r-min_occ_%r-min_files_%r" % (
        subset, min_occ, min_files)

    #   look for the cached processed data, return if present
    file_name = os.path.join(dir, name_base + ".pkl")
    data = util.try_pickle_load(file_name)
    if data is not None:
        return data

    #   did not find cached data, will have to process it
    #   log the loading process also to a file
    log_name = os.path.join(dir, name_base + ".log")
    log.addHandler(logging.FileHandler(log_name))

    #   process the data, cache it and return
    data = load_spacy_raw(subset, min_occ, min_files)
    util.try_pickle_dump(data, file_name)
    return data


def load_spacy_raw(subset, min_occ=1, min_files=1):
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
    as returned by the process_string() function.

    :param subset: The number of training files to process.
        If None (default), all of the files are processed.
    :param min_occ: Miniumum required number of occurences of
        a token (word) required for it to be included in the vocabulary.
        Default value (1) uses all words that occured in the trainset.
    :param min_files: Minumu required number of files a term has to
        occur in for it to be included in the vocabulary.
    """

    log.info("Processing data, trainset subset size :%r, minimum %r"
             " term occurences in a minimum of %r files",
             subset, min_occ, min_files)

    #   process questions
    log.info("Processing questions file")
    with open(os.path.join("data", "questions.txt")) as q_file:
        questions = [l.strip() for l in q_file.readlines()]

    #   remove question numbering and brackets from questions
    questions = map(lambda s: s[s.find(" ") + 1:], questions)
    questions = map(lambda s: re.sub(r"[\[\]]", "", s), questions)
    log.debug("Questions:")
    map(log.debug, questions)

    #   preprocess, tokenize and parse questions
    questions = map(process_string, questions)

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
    train_paths = os.listdir(train_dir)
    if subset is not None:
        log.info("Taking a subset of %d training files", subset)
        train_paths = train_paths[:subset]
    train_files = map(load_train_file, train_paths)

    #   vocabulary reduction, if desired
    if (min_occ + min_files) > 0:
        log.info("Reducing vocabulary, required min %d term occurences "
                 "across min %d files", min_occ, min_files)

        #   occurence counters for terms and lemmas
        voc_len = max([tf[0].max() for tf in train_files]) + 1
        lem_len = max([tf[1].max() for tf in train_files]) + 1
        voc_count = np.zeros(voc_len, dtype='uint32')
        lem_count = np.zeros(lem_len, dtype='uint32')
        file_voc_count = np.zeros(voc_len, dtype='uint32')
        file_lem_count = np.zeros(lem_len, dtype='uint32')

        def count(voc, lem):
            """
            Counts occurences of vocabulary indices in 'voc' and
            adds that count to the total count 'voc_count'. Does
            the same for lemmas. Also adds a +1 count to 'file_voc_count'
            for all vocabulary indices in 'inds'.
            """
            _voc_count = np.histogram(voc, voc_len, (-0.5, voc_len - 0.5))[0]
            voc_count.__iadd__(_voc_count)

            _lem_count = np.histogram(lem, lem_len, (-0.5, lem_len - 0.5))[0]
            lem_count.__iadd__(_lem_count)

            file_voc_count[voc] += 1
            file_lem_count[lem] += 1

        #   count occurences in train files
        map(count, [tf[0] for tf in train_files],
            [tf[1] for tf in train_files])

        #   term and lemma indices that should be kept
        voc_to_keep = np.arange(voc_len, )[
            np.logical_and(voc_count >= min_occ, file_voc_count >= min_files)]
        lem_to_keep = np.arange(lem_len, )[
            np.logical_and(lem_count >= min_occ, file_lem_count >= min_files)]
        new_voc_len = voc_to_keep.size
        new_lem_len = lem_to_keep.size
        log.info("New vocab len: %d, lemma len: %d", new_voc_len, new_lem_len)
        log.info("Tokens kept: %.2f",
                 float(voc_count[voc_to_keep].sum()) / float(voc_count.sum()))

        #   data structure for vocab and lemma conversion
        old_to_new_voc = np.ones((voc_len, ), dtype='uint32') * new_voc_len
        old_to_new_voc[voc_to_keep] = np.arange(voc_to_keep.size, )
        old_to_new_lem = np.ones((lem_len, ), dtype='uint32') * new_lem_len
        old_to_new_lem[lem_to_keep] = np.arange(lem_to_keep.size, )

        log.info("Converting trainset and questions to new vocabulary")

        def convert(tokens):
            tokens[0][:] = old_to_new_voc[tokens[0]]
            tokens[1][:] = old_to_new_lem[tokens[1]]
        map(convert, [tf for tf in train_files])
        map(convert, [q for q in questions])

    #   group sentences of a single question into a single array
    question_groups = [questions[i * 5: (i + 1) * 5] for
                       i in xrange(len(questions) / 5)]

    log.info("Loading done, returning data")
    return train_files, question_groups, answers


def ngrams(n, features, parent_ind=None, invalid_tokens={}):
    """
    Converts a text represented with feature indices inton-grams.
    Can create dependeny-syntax-tree-based n-grams and linear n-grams.

    All the features are appended for each of the terms in the ngram.
    For example, for n=2 (bigrams), and three features, the first resulting
    bigram would be [ ftrs[0][1], ftrs[1][1], ftrs[2][1], ftrs[0][0],
        ftrs[1][0], ftrs[2][0]]. Note that terms in n-grams are ordered
    last-to-first, while features are ordered first to last.

    Tree-based n-grams are formed by following the tree
    heirarchy upwards from a node. Dependency type can also be
    inserted into the n-gram as one of the features.

    Linear n-grams are generated if the 'parent_ind' parameter is
    None, otherwise tree-based n-grams are made.

    :param n: Desired n-gram length.
    :param features: An iterable of features. Each feature is a numpy
        array of indices. All of the features must have the same
        shape, that being (N, 1), where N is the number of terms in text.
    :param parent_ind: Indices of node parents in the dependancy syntax
        tree. If a node has no parent (is root of a sentence), then it's
        parent_ind is the same as it's own index.
    :param invalid_tokens: An iterable of token indices that should be
        removed from the resulting n-gram list. Useful for removing
        stop-words, substitute tokens etc.

    :return: Returns a numpy array of shape (count, features),
        where count is the number of resulting n-grams (depends on n),
        and features = n * len(features). The first len(features)
        columns in the result are the conditioned term, the following
        len(features) colums are the closest conditioning term, etc.
    """
    #   some info we will use
    token_len = features[0].size
    feature_count = len(features)
    use_tree = parent_ind is not None

    #   ensure feature and parent dimensions correspond
    for ftr in features:
        assert ftr.size == token_len, "All features must be of same size"
    if use_tree:
        assert parent_ind.size == token_len, \
            "Parent indices size not equal to feature size"

    #   calculate the shape of the resulting array
    shape = (token_len - (0 if use_tree else n - 1), n * feature_count)

    #   init array, and set first column (the conditioned term)
    r_val = np.zeros(shape, dtype='uint32')

    #   populate r_val with n-gram features
    #   iterate through all the features first
    for feature_ind, feature in enumerate(features):

        if use_tree:
            #   in tree-based-ngrams we go upwards through the tree
            #   for each features we will modify the parent_ind as we go along
            #   so ensure that the original parent_ind remains intact
            current_parent_ind = np.array(parent_ind)
            #   go through terms in the ngram for current feature
            for term_ind in xrange(n):
                #   set r_val values to current feature
                r_val[term_ind * feature_count + feature_ind, :] = feature
                #   step up through the tree-ngram heirarchy
                feature = feature[current_parent_ind]
                current_parent_ind = current_parent_ind[current_parent_ind]

        else:
            #   go through terms in the ngram for current feature
            for term_ind in xrange(n):
                #   linear n-grams are based on a backward sliding window
                feature = feature[n - 1 - term_ind:token_len - term_ind]
                r_val[term_ind * feature_count + feature_ind, :] = feature

    #   remove n-grams that contain invalid tokens
    if invalid_tokens is not None and len(invalid_tokens) > 0:

        #   a vector that contains invalid values for each column
        vec = np.ones(feature_count, dtype='uint32') * -1
        vec[invalid_tokens.keys()] = invalid_tokens.values()
        vec = np.tile(vec, n)

        #   locate invalid values, find rows that have any invalid values
        #   and remove those rows from r_val
        r_val = r_val[np.logical_not((r_val == vec).any(axis=1))]

    return r_val


def main():
    """
    Performs dataset loading / caching and prints out some
    information about the dataset.

    Allowed cmd-line flags:
        -s TS_FILES: Uses the reduced trainsed (TS_FILES trainset files)
        -o MIN_OCCUR: Only uses terms that occur MIN_OCCUR or more times
            in the trainset. Other terms are replaced with a special token.
        -f MIN_FILES: Only uses terms that occur in MIN_FILES or more files
            in the trainset. Other terms are replaced with a special token.
    """

    logging.basicConfig(level=logging.INFO)

    train_files, questions, answers = load_spacy(
        util.argv('-s', None, int),
        util.argv('-o', 1, int),
        util.argv('-f', 1, int)
    )

    log.info("Number of works in trainset: %d", len(train_files))
    log.info("Trainset token count: %d", sum([len(x[0]) for x in train_files]))
    log.info("Number of questions: %d", len(questions))

if __name__ == "__main__":
    main()
