"""
Implementation of a log-bilinear language model,
based on (Mnih, 2008.).
"""
import logging
from time import time

import numpy as np
import theano
import theano.tensor as T

import grad_descent

log = logging.getLogger(__name__)


class LLBL2():

    def __init__(self, n, vocab_size, repr_size, rng=None):

        log.info("Creating an LLBL, n=%d, vocab_size=%r, repr_size=%r",
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
            value=numpy_rng.uniform(-0.01, 0.01, size=(
                vocab_size, repr_size)).astype(theano.config.floatX),
            name='embedding', borrow=True)

        #   if weights are not provided, initialize them
        self.w = theano.shared(value=np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (vocab_size + (n - 1) * repr_size)),
                high=4 * np.sqrt(6. / (vocab_size + (n - 1) * repr_size)),
                size=((n - 1) * repr_size, repr_size)
            ),
            dtype=theano.config.floatX
        ), name='w', borrow=True)

        #   representation biases
        self.b_repr = theano.shared(
            value=np.zeros(repr_size, dtype=theano.config.floatX),
            name='b_repr',
            borrow=True
        )

        #   word biases
        self.b_word = theano.shared(
            value=np.zeros(vocab_size, dtype=theano.config.floatX),
            name='b_word',
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

        #   use linear composition of the input
        composition = T.dot(input_repr, self.w) + self.b_repr
        #   calculate corelation with the "output"
        correlation = T.dot(composition, emb.T) + self.b_word
        #   exponantiate to make it log-bilienar
        correlation = T.exp(correlation)

        #   define probability and cost
        self.probability = T.nnet.softmax(correlation)[
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
            "b_repr": self.b_repr,
            "b_word": self.b_word,
        }

        if not symbolic:
            r_val = dict([(k, v.get_value(borrow=True))
                          for k, v in r_val.iteritems()])

        return r_val

    def train(self, x_train, x_valid, mnb_size, epochs, eps, alpha,
              steps=1, weight_cost=1e-4):
        """
        Trains the LLBL with the given data. Returns a tuple containing
        (costs, times, hid_unit_activation_histograms). All three
        elements are lists, one item per epoch except for 'times' that
        has an extra element (training start time).

        :param x_train: Trainset of (N, n_vis) shape, where N is the
            number of samples.
        :param x_valid: Validation set of (N, n_vis) shape, where N is the
            number of samples.
        :param mnb_size: Minibatch size, the number of samples in the
            minibatch.
        :param epochs: Number of epochs (int) of training.
        :param eps: Learning rate. Either a float value to be
            used directly, or a callable that determines the
            learning rate based on epoch number and a list of
            error rates.
        :param alpha: float in range [0, 1]. Probability distribution
            (LLBL) learning is multiplied with alpha, while representation
            learning (word-vectors) is multiplied with (1 - alpha).
        :param steps: The number of steps to be used in PCD.
            Integer or callable, or a callable that determines the
            learning rate based on epoch number and a list of
            error rates.
        :param weight_cost: Regularization cost for L2 regularization
            (weight decay).
        """

        log.info('Training LLBL, epochs: %d, eps: %r, alpha: %.2f',
                 epochs, eps, alpha)

        #   pack trainset into a shared variable
        train_mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
        valid_mnb_count = (x_valid.shape[0] - 1) / mnb_size + 1
        x_train = theano.shared(x_train, name='x_train', borrow=True)
        x_valid = theano.shared(x_valid, name='x_valid', borrow=True)

        #   *** Creating a function for training the net
        #   first calculate CD "gradients"
        param_updates = grad_descent.gradient_updates_rms(
            self.cost_l2, self.params(True).values(), eps, 0.9)

        #   finally construct the function that updates parameters
        index = T.iscalar()
        train_f = theano.function(
            [index, self.l2_lmbd],
            self.cost,
            updates=param_updates.updates,
            givens={
                self.input: x_train[index * mnb_size: (index + 1) * mnb_size]
            }
        )
        #   ***  Done creating a function for training the net ###

        #   a separate function we will use for validation
        validate_f = theano.function(
            [index],
            self.cost,
            givens={
                self.input: x_valid[index * mnb_size: (index + 1) * mnb_size]
            }
        )

        #   things we'll track through training, for reporting
        train_costs = []
        valid_costs = []
        train_times = []

        #   iterate through the epochs
        log.info("Starting training")
        #   after each epoch, call a callback if provided
        epoch_callback = getattr(self, "epoch_callback", lambda a, b: None)
        mnb_callback = getattr(self, "mnb_callback", lambda a, b, c: None)
        epoch_callback(self, -1)
        mnb_callback(self, -1, -1)
        for epoch_ind, epoch in enumerate(range(epochs)):
            epoch_t0 = time()

            #   iterate learning through the minibatches
            def mnb_train(batch_ind):
                cost = train_f(batch_ind, weight_cost)
                log.debug('Mnb %d train cost %.5f', batch_ind, cost)
                mnb_callback(self, epoch_ind, batch_ind)
                return cost
            train_costs.append(np.mean(map(mnb_train, range(train_mnb_count))))

            valid_costs.append(
                np.mean(map(validate_f, range(valid_mnb_count))))
            epoch_callback(self, epoch_ind)
            train_times.append(time() - epoch_t0)

            log.info('Epoch %d:\n\ttrain cost: %.5f\n\tvalid cost: %.5f'
                     '\n\tduration %.2f sec', epoch_ind,
                     train_costs[-1], valid_costs[-1], train_times[-1]
                     )

        log.info('Training duration %.2f min', (sum(train_times)) / 60.0)

        return train_costs, valid_costs, train_times

    def __getstate__(self):
        """
        we are overriding pickling to avoid pickling
        any CUDA stuff, that will make our pickles GPU
        dependent.
        """

        raise "Implement me!"

    def __setstate__(self, state):
        """
        we are overriding pickling to avoid pickling
        any CUDA stuff, that will make our pickles GPU
        dependent.
        """

        raise "Implement me!"
