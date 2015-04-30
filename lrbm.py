"""
Implementation of a Restricted Boltzmann Machine.

Uses Theano, but does not utilize automatic gradient
calculation based on free energy (like in the Theano
RBM tutorial), but instead uses already defined CD
and PCD expressions.
"""
import numpy as np
import theano
import theano.tensor as T
import logging
from time import time

log = logging.getLogger(__name__)


class LRBM():

    """
    Language modeling RBM. Embeds a normal RBM but also uses
    a vocabulary matrix that maps words into d-dimensional
    vectors.
    """

    def __init__(self, n, vocab_size, repr_size, n_hid, rng=None):

        log.info("Creating an LRBM, n=%d, vocab_size=%r, repr_size=%r"
                 "n_hid=%d", n, vocab_size, repr_size, n_hid)

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
            name='repr_embedding', borrow=True)

        #   figure out how many visible variables there are
        _n_vis = n * repr_size

        #   if weights are not provided, initialize them
        self.w = theano.shared(value=np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_hid + _n_vis)),
                high=4 * np.sqrt(6. / (n_hid + _n_vis)),
                size=(_n_vis, n_hid)
            ),
            dtype=theano.config.floatX
        ), name='w', borrow=True)

        #   if hidden biases are not provided, initialize them
        self.b_hid = theano.shared(
            value=np.zeros(n_hid, dtype=theano.config.floatX),
            name='b_hid',
            borrow=True
        )

        #   if visible biases are not provided, initialize them
        self.b_vis = theano.shared(
            value=np.zeros(_n_vis, dtype=theano.config.floatX),
            name='b_vis',
            borrow=True
        )

        #   initialize theano symbolic variables
        self.init_theano_vars()

    def init_theano_vars(self):
        """
        Initializes Theano expressions used in training and evaluation.
        """

        #   input is a matrix of (N, n * len(repr_size)) dimensionality
        #   each row represents an n-gram
        input = T.matrix('input', dtype='uint16')
        self.input = input
        emb = self.embedding

        #   a sym var for input mapped to emebedded representations
        input_repr = emb[input].flatten(2)
        self.input_repr = input_repr

        #   activation probability, hidden layer, positive phase
        self.hid_prb_pos = T.nnet.sigmoid(
            T.dot(input_repr, self.w) + self.b_hid)

        #   binary activation, hidden layer, positive phase
        self.hid_act_pos = self.theano_rng.binomial(
            n=1, p=self.hid_prb_pos, size=self.hid_prb_pos.shape,
            dtype=theano.config.floatX)

        #   activation, visible layer, negative phase
        #   only the conditioned term gets updated
        _vis_neg = T.nnet.sigmoid(
            T.dot(self.hid_act_pos,
                  self.w[:self.repr_size].T) + self.b_vis[:self.repr_size])
        #   but we need the whole visible vector, for the updates
        self.vis_neg = T.concatenate(
            (_vis_neg, input_repr[:, self.repr_size:]), axis=1)

        #   a function that returns the energy symbolic variable
        #   given visible and hidden unit symbolic variables
        def energy(visible, hidden):
            return -T.dot(visible, self.b_vis.T) \
                - T.dot(hidden, self.b_hid.T) \
                - (T.dot(visible, self.w) * hidden).sum(axis=1)

        #   activation probability, hidden layer, negative phase
        self.hid_prb_neg = T.nnet.sigmoid(
            T.dot(self.vis_neg, self.w) + self.b_hid)

        #   standard energy of the input
        self.energy = energy(input_repr, self.hid_prb_pos)

        #   exact probablity of a single sample, given model params only
        #   first define a function that calculates the prob. of one sample
        def _probability(sample):
            #   a matrix with input-representations for all words,
            #   conditioned on the conditioning terms of the sample
            _partition = T.concatenate([
                emb,
                T.tile(emb[sample[1:]].flatten().dimshuffle(('x', 0)),
                       (self.vocab_size, 1))],
                axis=1).astype('float64')
            #   input to each hidden unit, for every input-repr in _partition
            _hid_in = -T.dot(_partition, self.w) - self.b_hid
            #   a binom signifying a hidden unit being on or off
            _hid_in_exp = (1 + T.exp(_hid_in))
            #   divide with mean for greater numeric stability
            #   does not change end probability
            _hid_in_exp /= _hid_in_exp.mean()
            _probs = _hid_in_exp.prod(axis=1)
            _probs *= -T.exp(T.dot(_partition, self.b_vis))
            return _probs[sample[0]] / _probs.sum()
        #   now use theano scan to calculate probabilities for all inputs
        self.probability, _ = theano.scan(_probability,
                                          outputs_info=None,
                                          sequences=[input])

        #   distribution of the vocabulary, given hidden state
        #   first define a function that calcs the distribution
        #   given a single hidden_activation sample
        def _distribution_w(vis_state, hid_act):
            _vis = T.concatenate([
                emb,
                T.tile(emb[vis_state[1:]].flatten().dimshuffle(('x', 0)),
                       (self.vocab_size, 1))],
                axis=1).astype('float64')
            _energies = energy(_vis, T.tile(hid_act, (self.vocab_size, 1)))
            _energies -= _energies.min() + 400
            _probs = T.exp(-_energies)
            return _probs / _probs.sum()
        #   now calculate probability distributions for the whole input
        self.distribution_w_given_h = theano.scan(
            _distribution_w,
            outputs_info=None,
            sequences=[input, self.hid_act_pos])

        #   reconstruction error that we want to reduce
        #   we use contrastive divergence to model the distribution
        #   and optimize the vocabulary
        self.reconstruction_error = (
            (input_repr - self.vis_neg) ** 2).mean()

    def mean_log_lik(self, x):
        """
        Calculates the mean log-loss of the given samples.
        Note that calculation complexity is O(v1 * v2 * v3 * ...),
        where vX is vocabulary size of feature X, for each used feature.

        :param x: Samples of shape (N, n * len(used_ftrs)).
        :return: Mean log loss.
        """
        energy_f = theano.function([self.input], self.energy)

        def _probability(sample):

            #   create samples for each vocab word
            #   given the conditioning part of the sample
            _partition = np.hstack((
                np.arange(self.vocab_size, dtype=sample.dtype).reshape(
                    self.vocab_size, 1),
                np.tile(sample[1:], (self.vocab_size, 1))))
            #   calculate their energy, normalize, and exp
            _partition_en = energy_f(_partition).astype('float32')
            _partition_en -= _partition_en.min() + 30
            _partition_exp = np.exp(-_partition_en)
            return _partition_exp[sample[0]] / _partition_exp.sum()

        _probs = map(_probability, x)
        return np.mean(np.log(_probs))

    def train(self, x_train, x_valid, mnb_size, epochs, eps, alpha,
              steps=1, weight_cost=1e-4):
        """
        Trains the RBM with the given data. Returns a tuple containing
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
            (RBM) learning is multiplied with alpha, while representation
            learning (word-vectors) is multiplied with (1 - alpha).
        :param steps: The number of steps to be used in PCD.
            Integer or callable, or a callable that determines the
            learning rate based on epoch number and a list of
            error rates.
        :param weight_cost: Regularization cost for L2 regularization
            (weight decay).
        """

        log.info('Training RBM, epochs: %d, eps: %r, steps:%d',
                 epochs, eps, steps)

        #   pack trainset into a shared variable
        mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
        x_train = theano.shared(x_train, name='x_train', borrow=True)

        #   *** Creating a function for training the net

        #   first calculate CD "gradients"
        vis_pos = self.input_repr
        vis_neg = self.vis_neg
        hid_pos = self.hid_prb_pos
        hid_neg = self.hid_prb_neg
        grad_b_vis = vis_pos.mean(axis=0) - vis_neg.mean(axis=0)
        grad_b_hid = hid_pos.mean(axis=0) - hid_neg.mean(axis=0)
        grad_w = (T.dot(vis_pos.T, hid_pos) - T.dot(vis_neg.T, hid_neg)
                  ) / T.cast(vis_pos.shape[0], theano.config.floatX)

        #   calculate the "gradient" for word embedding modification
        grad_l = T.dot(hid_pos, self.w.T) - T.dot(hid_neg, self.w.T)
        #   reorganize the grad_l from (N, n * d) to (N * n, d)
        grad_l = [grad_l[:, i * self.repr_size: (i + 1) * self.repr_size]
                  for i in xrange(self.n)]
        grad_l = T.concatenate(grad_l, axis=0)
        #   add embedding optimization to updates
        #   reorganize input from (N, n) into (N * n)
        input_stack = self.input.dimshuffle((1, 0)).flatten()

        #   add regularization to gradients
        grad_b_vis -= weight_cost * self.b_vis
        grad_b_hid -= weight_cost * self.b_hid
        grad_w -= weight_cost * self.w

        #   define a list of updates that happen during training
        eps_th = T.scalar("eps", dtype=theano.config.floatX)
        updates = [
            (self.w, self.w + eps_th * alpha * grad_w),
            (self.b_vis, self.b_vis + eps_th * alpha * grad_b_vis),
            (self.b_hid, self.b_hid + eps_th * alpha * grad_b_hid),
            (self.embedding, T.inc_subtensor(
                self.embedding[input_stack], eps_th * (1 - alpha) * grad_l))
        ]

        #   finally construct the function that updates parameters
        index = T.iscalar()
        train_f = theano.function(
            [index, eps_th],
            self.reconstruction_error,
            updates=updates,
            givens={
                self.input: x_train[index * mnb_size: (index + 1) * mnb_size]
            }
        )
        #   ***  Done creating a function for training the net ###

        #   a separate function we will use for validation
        validate_f = theano.function(
            [self.input],
            self.reconstruction_error,
        )

        #   things we'll track through training, for reporting
        train_costs_mnb = []
        train_costs_ep = []
        valid_costs_ep = []
        train_times_ep = []

        #   iterate through the epochs
        log.info("Starting training")
        #   after each epoch, call a callback if provided
        epoch_callback = getattr(self, "epoch_callback", lambda a, b: None)
        mnb_callback = getattr(self, "mnb_callback", lambda a, b, c: None)
        epoch_callback(self, -1)
        mnb_callback(self, -1, -1)
        for epoch_ind, epoch in enumerate(range(epochs)):
            epoch_t0 = time()

            #   calc epsilon for this epoch
            if not isinstance(eps, float):
                epoch_eps = eps(epoch_ind, train_costs_ep)
            else:
                epoch_eps = eps

            #   iterate through the minibatches
            for batch_ind in xrange(mnb_count):
                train_costs_mnb.append(train_f(batch_ind, epoch_eps))
                log.debug('Mnb %d train cost %.5f',
                          batch_ind, train_costs_mnb[-1])
                mnb_callback(self, epoch_ind, batch_ind)

            train_costs_ep.append(
                np.array(train_costs_mnb)[-mnb_count:].mean())
            valid_costs_ep.append(validate_f(x_valid))
            train_times_ep.append(time() - epoch_t0)

            log.info(
                'Epoch %d:\n\ttrain cost: %.5f\n\tvalid cost: %.5f'
                '\n\tduration %.2f sec',
                epoch_ind,
                train_costs_ep[-1],
                valid_costs_ep[-1],
                train_times_ep[-1]
            )
            epoch_callback(self, epoch_ind)

        log.info('Training duration %.2f min',
                 (sum(train_times_ep)) / 60.0)

        return train_costs_ep, train_times_ep

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
