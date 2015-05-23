"""
Implementation of feed-forward neural net for
language modeling.
"""

import logging

import numpy as np
import theano
import theano.tensor as T

from iterative_model import IterativeModel

log = logging.getLogger(__name__)


class LNNet(IterativeModel):
    """
    A neural net for language modeling. No hidden layers,
    just input and output. Uses distributed representations
    which are trained alongside the net.
    """

    def __init__(self, n, vocab_size, repr_size, rng=None):

        log.info("Creating an LMLP, n=%d, vocab_size=%r, repr_size=%r",
                 n, vocab_size, repr_size)

        #   n-gram size
        self.n = n

        #   random number generators
        if rng is None:
            numpy_rng = np.random.RandomState()
        elif isinstance(rng, int):
            numpy_rng = np.random.RandomState(rng)
        else:
            numpy_rng = rng
        self.theano_rng = T.shared_randomstreams.RandomStreams(
            numpy_rng.randint(2 ** 30))

        #   create word embedding and corresponding info
        self.repr_size = repr_size
        self.vocab_size = vocab_size
        self.embedding = theano.shared(
            value=numpy_rng.uniform(-1e-3, 1e-3, size=(
                vocab_size, repr_size)).astype(theano.config.floatX),
            name='embedding', borrow=True)

        #   if weights are not provided, initialize them
        self.w = theano.shared(value=np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (vocab_size + (n - 1) * repr_size)),
                high=4 * np.sqrt(6. / (vocab_size + (n - 1) * repr_size)),
                size=((n - 1) * repr_size, vocab_size)
            ),
            dtype=theano.config.floatX
        ), name='w', borrow=True)

        #   if hidden biases are not provided, initialize them
        self.b_out = theano.shared(
            value=np.zeros(vocab_size, dtype=theano.config.floatX),
            name='b_out',
            borrow=True
        )

        #   initialize theano symbolic variables
        self.init_theano_vars()

    def init_theano_vars(self):
        """
        Initializes Theano expressions used in training and evaluation.
        """

        #   input is a matrix of (N, n) dimensionality
        #   each row represents an n-gram
        input = T.matrix('input', dtype='uint16')
        self.input = input
        emb = self.embedding

        #   a sym var for input mapped to emebedded representations
        input_repr = emb[input[:, 1:]].flatten(2)
        self.input_repr = input_repr

        output = T.dot(input_repr, self.w) + self.b_out

        self.probability = T.nnet.softmax(output)[
            T.arange(input.shape[0]), input[:, 0]]
        self.cost = -T.log(self.probability).mean()

        #   also define a variable for L2 regularized cost
        self.l2_lmbd = T.fscalar('weight_decay_lambda')
        self.cost_l2 = self.cost + self.l2_lmbd
        for param in [self.embedding, self.w]:
            self.cost_l2 += self.l2_lmbd * (param ** 2).sum()

    def params(self, symbolic=False):
        """
        Returns a dictionary of all the model parameters. The dictionary
        keys are parameter names, values are parameter.

        :param symbolic: If symbolic Theano variables should be returned.
            If False (default), then their numpy value arrays are returned.
        :return: A dictionary that maps parameter names to their values
            (shared theano variables, or numpy arrays).
        """
        r_val = {
            "w": self.w,
            "embedding": self.embedding,
            "b_out": self.b_out,
        }

        if not symbolic:
            r_val = dict([(k, v.get_value(borrow=True))
                          for k, v in r_val.iteritems()])

        return r_val
