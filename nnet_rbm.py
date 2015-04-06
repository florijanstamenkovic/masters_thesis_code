"""
Module for evaluating the RBM energy-based neural net
language models on the Microsoft
Sentences Completion Challenge dataset (obtained through
the 'data' module).
"""

import logging

log = logging.getLogger(__name__)


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
