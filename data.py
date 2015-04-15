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
from time import time
import sys


log = logging.getLogger(__name__)


def preprocess_string(string):
    """
    Does string preprocessing. Typically this means converting
    multiple spaces to single spaces, digits to special number
    tokens etc. Returns the processed string.

    param string: The string to process.
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

    return string


def vocab_to_ind(term, vocab):
    ind = vocab.get(term, None)
    if ind is None:
        ind = len(vocab)
        vocab[term] = ind
    return ind


#   dictionaries that map terms to unique indices
#   used in the process_string method
_orth_to_ind = {}
_lemma_to_ind = {}
_lemma_4_to_ind = {}


def process_string(string, preprocess=True):
    """
    Processes a string of text.
    Extracts and returns vocabulary indices,
    lemma indices (lemm), lemma-shortened-to-N indices (lemm_N),
    part of speech indices (pos), detailed part of speech tag
    indices (tag), syntax to-parent-dependency type indices
    (dep), syntax-parent-indices (head) and sentence indices
    (sent).

    :param string: A string of text.
    :param preprocess: If or not preprocessing should be done
        on the text.
    :return: A single numpy array that has as many rows as there
        are tokens in the processed string, and columns are:
        (orth, lemm, lemm_4, pos, tag, dep_type, head, sent).
    """

    #   lazy import / nlp pipeline creation, to make the spacy dependancy
    #   optinal (only needed in this function)
    import spacy.en
    nlp_key = "_nlp_key_in_globals_"
    if nlp_key not in globals().keys():
            #   the tokenizer / parser used
        nlp = spacy.en.English()
        globals()[nlp_key] = nlp
    nlp = globals()[nlp_key]

    if preprocess:
        string = preprocess_string(string)

    #   tokenization, only accepts unicode, so convert
    if isinstance(string, str):
        string = unicode(string)
    tokens = nlp(string, True, True)

    #   return value is of shape (token_count, feature_count + 2)
    #   where last 2 columns are syntax-parent-indices and sentence indices
    r_val = np.zeros((len(tokens), 8), dtype='uint32')

    #   extract features
    r_val[:, 0] = [vocab_to_ind(t.orth_.lower(),
                                _orth_to_ind) for t in tokens]
    r_val[:, 1] = [vocab_to_ind(t.lemma_.lower(),
                                _lemma_to_ind) for t in tokens]
    r_val[:, 2] = [vocab_to_ind(t.lemma_[:4].lower(),
                                _lemma_4_to_ind) for t in tokens]
    r_val[:, 3] = [t.pos for t in tokens]
    r_val[:, 4] = [t.tag for t in tokens]
    r_val[:, 5] = [t.dep for t in tokens]

    #   convert head references to indices
    indices = dict(zip(tokens, range(len(tokens))))
    r_val[:, 6] = [indices[t.head] for t in tokens]

    #   mark separate sentences
    #   we need consistency with syntax-trees, so base sentence
    #   segmentation on that
    _sents = np.array(r_val[:, 6])
    _sents_new = _sents[_sents]
    #   step upwards through parent indices while they keep changing
    while (_sents != _sents_new).any():
        _sents = _sents_new
        _sents_new = _sents[_sents]
    #   np.unique can give a reconstruction of root-node marks
    #   these are our sentence indices
    r_val[:, 7] = np.unique(_sents, return_inverse=True)[1]

    return r_val


def load(subset=None, min_occ=1, min_files=1):
    """
    Loads the raw text data that constitutes the Microsoft
    Sentence Completion Challenge (stored in ./data/).
    Processes the data, tokenizes and parses it, and returns
    the results.

    Returned is a tuple (train_sents, question_groups, answers,
    feature_sizes).
    The 'train_sents' numpy array of shape (token_count, feature_count).
    Features colums are at first textual (orth, lemma, lemma_4),
    then syntactic (pos, dependency-type). The [-2] column contains
    syntactic-parent-indices, and the [-1] column denotes to which
    sentence the token belongs. The 'question_groups' object
    is an iterable question groups. Each group consists of 5
    sentences (one of which is correct). Each sentence is a
    parsed-text-array as described above. The 'answers'
    object is a numpy array of shape (question_group_count, )
    that contains the indices of the correct sentences in
    question groups. The 'feature_sizes' object is a numpy
    array contaning the dimensionality of each feature.

    :param subset: The number of training files to process.
        If None (default), all of the files are processed.
    :param min_occ: Miniumum required number of occurences of
        a token (word) required for it to be included in the vocabulary.
        Default value (1) uses all words that occured in the trainset.
    :param min_files: Minumum required number of files a term has to
        occur in for it to be included in the vocabulary.
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
    data = _load(subset, min_occ, min_files)
    util.try_pickle_dump(data, file_name)
    return data


def _load(subset=None, min_occ=1, min_files=1):
    """
    Private function that does the work for 'load',
    which effectively decorates this function with
    data caching capability.
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
    questions = [process_string(q, False) for q in questions]

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
        etc. The text is then processed using the 'process'
        function, and sentence-segmented.

        :param train_file: Path to training file.
        :return: The same as 'process' function.
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

        tokens = process_string(data)
        log.info("Training file has %d tokens in %d sentences",
                 tokens.shape[0], np.unique(tokens[:, 7]).size)
        return tokens

    log.info("Processing training data")
    train_dir = os.path.join("data", "trainset")
    train_paths = sorted(os.listdir(train_dir))
    if subset is not None:
        log.info("Taking a subset of %d training files", subset)
        train_paths = train_paths[:subset]

    #   sentences grouped by training file, and also merged into a single list
    tf_tokens = map(load_train_file, train_paths)

    #   we want to group sentences from all training files
    #   into a single numpy array
    #   to do that we need to ensure that sentences indices
    #   are non-repetitive
    _sentences_processed = 0
    for tf in tf_tokens:
        tf[:, -1] += _sentences_processed
        _sentences_processed = tf[:, -1].max() + 1

    #   we also want to ensure that syntax-parent indices are corret
    _tokens_processed = 0
    for tf in tf_tokens:
        tf[:, -2] += _tokens_processed
        _tokens_processed += tf.shape[0]

    #   before grouping all the training files into a single np array
    #   do vocabulary reduction

    #   vocabulary reduction, if desired
    if min_occ > 1 or min_files > 1:
        log.info("Reducing vocabulary, required min %d term occurences "
                 "across min %d files", min_occ, min_files)

        #   a list of indices of features in each training file we
        #   impose vocabulary limations on
        ftr_to_reduce = np.array([0, 1, 2])

        #   find out the max value of each feature across all training files
        ftr_lens = np.array([tf.max(axis=0) for tf in tf_tokens]).max(
            axis=0)[ftr_to_reduce] + 1
        log.info("Old feature sizes: %r", ftr_lens)

        #   create counter arrays for tokens of each feature
        ftr_counts = map(lambda len: np.zeros(len, dtype='uint32'), ftr_lens)
        #   create counter arrays for feature-in-file occurenecs
        file_ftr_counts = map(lambda len: np.zeros(len, dtype='uint16'),
                              ftr_lens)

        #   count the occurences for relevant features
        for i, ftr in enumerate(ftr_to_reduce):

            #   we are interested in per-file info, so iterate over files
            for tf in tf_tokens:

                #   count occurences of features in current tf
                tf_counts = np.histogram(
                    tf[:, ftr], np.arange(ftr_lens[i] + 1) - 0.5)[0]

                #   sum those occurences to total counts
                ftr_counts[i] += tf_counts
                file_ftr_counts[i] += tf_counts > 0

        #   feature indices that should be kept
        keep_f = lambda len, count, f_count: np.arange(len, )[np.logical_and(
            count >= min_occ, f_count >= min_files)]
        ftr_to_keep = map(keep_f, ftr_lens, ftr_counts, file_ftr_counts)

        #   determine the new feature sizes and number of tokens kept
        new_ftr_lens = np.array(
            map(lambda f: len(f) + 1, ftr_to_keep), dtype=int)
        tokens_kept = map(lambda count, keep: count[keep].sum() /
                          float(count.sum()), ftr_counts, ftr_to_keep)

        log.info("New feature sizes: %r", new_ftr_lens)
        log.info("Features kept: (%s)", ", ".join(
            ["%.4f" % f for f in tokens_kept]))

        #   data structures for token conversions
        def old_to_new_ftr_f(old_len, substitute, keep):
            r_val = np.ones(old_len, dtype='uint32') * substitute
            r_val[keep] = np.arange(keep.size)
            return r_val
        old_to_new_ftr = map(
            old_to_new_ftr_f, ftr_lens, new_ftr_lens - 1, ftr_to_keep)

        log.info("Converting trainset and questions to new vocabulary")

        def convert(tokens):
            for old_to_new, ind in zip(old_to_new_ftr, ftr_to_reduce):
                ftr = tokens[:, ind]
                ftr[:] = old_to_new[ftr]
        map(convert, tf_tokens)
        map(convert, questions)

    #   group sentences of a single question into a single array
    question_groups = [questions[i * 5: (i + 1) * 5] for
                       i in xrange(len(questions) / 5)]

    #   group the training files into a single numpy array
    trainset = np.concatenate(tf_tokens)

    #   get the final feature sizes
    feature_sizes = trainset.max(axis=0) + 1

    log.info("Loading done, returning data")
    return trainset, question_groups, answers, feature_sizes


def ngrams(n, tree, tokens, invalid_tokens=None):
    """
    Converts a text represented with tokens into n-grams.
    Can create dependeny-syntax-tree-based n-grams and linear n-grams.

    All the features are appended for each of the terms in the ngram.
    For example, for n=2 (bigrams), and three features, the first resulting
    bigram would be [ ftrs[0][1], ftrs[1][1], ftrs[2][1], ftrs[0][0],
        ftrs[1][0], ftrs[2][0]]. Note that terms in n-grams are ordered
    last-to-first, while features are ordered first to last.

    Tree-based n-grams are formed by following the tree
    heirarchy upwards from a node, so that the first term in the ngram
    is the node itself, the next is it's parent, etc.

    :param n: Desired n-gram length.
    :param tree: If or not tree ngrams should be generated.
    :param tokens: A numpy array of shape (N_tokens, feature_count).
    :param invalid_tokens: A map indicating which tokens should for which
        features be considered invalid. For example {0: 123, 2: 55} indicates
        that for the feature at index 0 token 123 is invalid, and for the
        feature at index 2 token 55 is invalid. All the n-grams containing
        invalid tokens are filtered out from the final results.

    :return: Returns a numpy array of shape (count, n_gram_terms),
        where count is the number of resulting n-grams,
        and n_gram_terms = n * len(features). The first len(features)
        columns in the result are the conditioned term, the following
        len(features) colums are the closest conditioning term, etc.
    """
    #   get syntax-parent-indices, sentence indices
    #   and make tokens variable see only features
    parent_inds = tokens[:, -2]
    sent_inds = tokens[:, -1]
    tokens = tokens[:, :-2]

    #   some more info about sentences
    #   we only need it for linear ngrams
    if not tree:
        #   sentence lengths
        _sent_inds, sent_lens = np.unique(sent_inds, return_counts=True)
        assert (_sent_inds == np.arange(_sent_inds.size)).all()

        #   sentence n-gram lengths
        sent_ngram_lens = np.maximum(
            sent_lens - n + 1, np.zeros(sent_lens.shape))

        #   ranges for sentences in features, and ngrams
        def ranges(lengths):
            ranges = np.zeros((lengths.size, 2), dtype='uint32')
            ranges[:, 1] = np.cumsum(lengths)
            ranges[1:, 0] = ranges[:-1, 1]
            return ranges
        sent_ranges = ranges(sent_lens)
        sent_ngram_ranges = ranges(sent_ngram_lens)

    #   total number of tokens and features
    token_count, feature_count = tokens.shape

    #   calculate the shape of the resulting array
    if tree:
        shape = (token_count, n * feature_count)
    else:
        shape = (sent_ngram_lens.sum(), n * feature_count)

    r_val = np.zeros(shape, dtype='uint16')

    #   populate r_val with n-gram features
    #   iterate through all the features first
    for feature_ind in xrange(feature_count):

        feature = tokens[:, feature_ind]

        if tree:
            #   in tree-based-ngrams we go upwards through the tree
            #   for each features we will modify the parent_inds as we go along
            #   so ensure that the original parent_inds remains intact
            current_parent_inds = parent_inds
            #   go through terms in the ngram for current feature
            for term_ind in xrange(n):
                #   set r_val values to current feature
                r_val[:, term_ind * feature_count + feature_ind] = feature
                #   step up through the tree-ngram heirarchy
                feature = feature[current_parent_inds]
                current_parent_inds = current_parent_inds[current_parent_inds]

        else:

            #   linear n-grams are slightly more touchy because we
            #   can't have ngrams across sentence boundaries
            for sent_ind, (sent_start, sent_end) in enumerate(sent_ranges):

                #   extract the sentence from the feature
                sent = feature[sent_start: sent_end]
                sent_len = sent_end - sent_start
                ngram_start, ngram_end = sent_ngram_ranges[sent_ind]

                #   go through terms in the ngram for current sentence
                for term_ind in xrange(n):
                    #   linear n-grams are based on a backward sliding window
                    ftr = sent[n - 1 - term_ind:sent_len - term_ind]
                    r_val[ngram_start: ngram_end,
                          term_ind * feature_count + feature_ind] = ftr

    #   remove n-grams that contain invalid tokens
    if invalid_tokens is not None and len(invalid_tokens) > 0:

        #   a vector that contains invalid values for each column
        vec = np.ones(feature_count, dtype='uint16') * -1
        vec[invalid_tokens.keys()] = invalid_tokens.values()
        vec = np.tile(vec, n)

        #   locate invalid values, find rows that have any invalid values
        #   and remove those rows from r_val
        ok_rows = np.logical_not((r_val == vec).any(axis=1))
        r_val = r_val[ok_rows]

    return r_val


def load_ngrams(n, features_use, tree, subset=None, min_occ=1, min_files=1,
                remove_subst_tokens=True):
    """
    Loads the dataset for microsot sentence completion challenge, processed
    into ngrams.

    The raw dataset is loadaed and processed using the 'load' function,
    to which 'subset', 'min_occ' and 'min_files' are forwared.

    The resulting dataset is then processed into ngrams using the
    'ngrams' function, to which 'n' and 'tree' parameter are forwarded.
    This is then cached on the disk for subsequent usage.

    The resulting ngrams are pruned from unwanted features as indicated
    by the 'features_use parameter'.

    Returns a tuple (sents, q_groups, answers, feature_sizes).
    This reflects the returned value by the 'load' function, except that
    'sents' and 'g_groups' are now not just features extracted from text,
    but ngrams built from those features.
    """

    features_use = np.array(features_use, dtype=bool)

    log.info("Loading %d-grams, %s, features_use: %s",
             n, "tree" if tree else "linear",
             "".join([str(int(i)) for i in features_use]))

    dir = os.path.join("data", "processed")
    if not os.path.exists(dir):
        os.makedirs(dir)
    name_base = "%s-%d_grams-subset_%r-min_occ_%r-min_files_%r" % (
        "tree" if tree else "linear", n, subset, min_occ, min_files)

    #   tree-grams can all be seen as a feature-subset of 4 grams
    if tree and n < 4:
        ngrams_all = load_ngrams(
            4, np.ones(features_use.size, dtype=bool), tree, subset,
            min_occ, min_files, remove_subst_tokens)
    else:
        #   look for the cached 4-grams with all the features
        file_name = os.path.join(dir, name_base + ".pkl")
        ngrams_all = util.try_pickle_load(file_name)
        #   it is possible that sentences are split
        #   in order to avoid Python bug with storing large arrays
        if ngrams_all is not None and isinstance(ngrams_all[0], list):
            sents = np.vstack(ngrams_all[0])
            ngrams_all = (sents,) + ngrams_all[1:]

    #   if unable to load cached data, create it
    if ngrams_all is None:
        #   load data
        tokens, q_groups, answers, ftr_sizes = load(subset, min_occ, min_files)

        #   tokens that should be present in ngrams
        #   the purpose is to remove ngrams containing tokens that are
        #   substitutes for removed ones
        invalid_tokens = None
        if remove_subst_tokens:
            invalid_tokens = dict(zip(range(3), ftr_sizes[:3] - 1))
            log.info("Invalid tokens: %r", invalid_tokens)

        #   define a function for generating ngrams, and process
        #   trainset and questions
        _ngrams = lambda tokens: ngrams(n, tree, tokens, invalid_tokens)
        sent_ngrams = _ngrams(tokens)
        q_ngrams = [map(_ngrams, qg) for qg in q_groups]

        #   store the processed data for subsequent usage
        #   split sent ngrams to avoid Py bug with pickling large arrays
        util.try_pickle_dump((
            np.vsplit(sent_ngrams, np.arange(10, ) * (len(sent_ngrams) / 10)),
            q_ngrams, answers, ftr_sizes), file_name)
        ngrams_all = (sent_ngrams, q_ngrams, answers, ftr_sizes)

    #   remove unwanted features from ngrams_all
    used_ftr = np.arange(ngrams_all[0].shape[1])[np.tile(features_use, n)]
    sents = ngrams_all[0][:, used_ftr]
    q_groups = [[q[:, used_ftr] for q in qg] for qg in ngrams_all[1]]

    return (sents, q_groups) + ngrams_all[2:]


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
        -n Also do n-gram extraction.
    """

    logging.basicConfig(level=logging.INFO)

    subset = util.argv('-s', None, int)
    min_occ = util.argv('-o', 1, int)
    min_files = util.argv('-f', 1, int)

    if '-n' in sys.argv:

        def extract_and_report(n, tree):
            """
            Extracts n-grams and prints out basic info,
            which caches them for subsequent usage.
            """
            t0 = time()
            res = load_ngrams(n, np.ones(6), tree, subset, min_occ, min_files)
            log.info("%s %d-grams, count=%d, extracted in %.2f seconds",
                     "Tree" if tree else "Linear", n,
                     res[0].shape[0], time() - t0)
            del res

        #   extract linear ngrams
        for n in range(1, 5):
            extract_and_report(n, False)
        extract_and_report(4, True)

    else:
        data = load(subset, min_occ, min_files)
        del data

if __name__ == "__main__":
    main()
