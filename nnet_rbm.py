"""
Module for evaluating the RBM energy-based neural net
language models on the Microsoft
Sentences Completion Challenge dataset (obtained through
the 'data' module).
"""

import logging
import data
import numpy as np

log = logging.getLogger(__name__)


def dataset_split(subset=None, min_occ=1, min_files=1, validation=0.05, test=0.05):

    log.info("Performing dataset split, validation size: %.2f, "
             "test size: %.2f", validation, test)

    #   load the training data
    train_files, question_groups, answers = data.load_spacy(
        subset, min_occ, min_files)

    #   count the total number of tokens
    train_files_sizes = np.array(
        [tf[0].size for tf in train_files], dtype='uint32')
    log.info("Trainset contains %d tokens", train_files_sizes.sum())

    #   generate info about which token belongs where
    #   0 - means trainset, 1 - means validation, 2 - means test
    token_set = map(lambda l: np.random.choice(
        3, l, p=[1. - validation - test, validation, test]),
        train_files_sizes)

    #   prepare numpy arrays for train, validation and test data
    _set_size = lambda s: sum([(t == s).size for t in token_set])
    _ftr_count = len(train_files[0])
    datasets = [np.zeros((_set_size(i), _ftr_count), dtype='uint32') for i in range(3)]
    log.info("Dataset sizes, train: %r, validation: %r, test: %r",
             datasets[0].shape, datasets[1].shape, datasets[2].shape)

    #   split data where it needs to go
    current_inds = [0, 0, 0]
    for tf, tf_assign in zip(train_files, token_set):
        counts = [tf_assign == i for i in range(3)]


def main():
    """
    Trains and evaluates RBM energy based neural net
    language models on the Microsoft Sentence Completion
    Challenge dataset.
    """
    logging.basicConfig(level=logging.INFO)
    log.info("RBM energy-based neural net language model")

if __name__ == '__main__':
    main()
