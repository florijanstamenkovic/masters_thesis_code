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

    def __init__(self, n, vocab_sizes, repr_sizes, n_hid, rng=None):

        log.info("Creating an LRBM, n=%d, vocab_sizes=%r, repr_sizes=%r",
                 n, vocab_sizes, repr_sizes)

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

        #   a function that initializes and returns
        #   a vocabulary embedding matrix
        def create_dim_var(vocab_size, repr_size):
            """
            :param vocab_size: Vocabulary length.
            :param repr_size: Dimensionality of the representation vector.
            """
            return theano.shared(
                value=numpy_rng.uniform(-1e-3, 1e-3, size=(
                    vocab_size, repr_size)).astype(theano.config.floatX),
                name='repr_embedding', borrow=True)

        #   create word embeddings and corresponding info
        self.embeddings = map(create_dim_var, vocab_sizes, repr_sizes)
        self.term_repr_size = sum(repr_sizes)
        #   calc range of each representation in a 1-gram vector
        _sizes_cs = np.hstack(([0], np.cumsum(repr_sizes)))
        self.repr_ranges = np.array(zip(_sizes_cs[:-1], _sizes_cs[1:]),
                                    dtype='int32')

        #   figure out how many visible variables there are
        _n_vis = n * self.term_repr_size

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
        self.init_vars()

    def init_vars(self):

        #   input is a matrix of (N, n * len(repr_sizes)) dimensionality
        #   each row represents an n-gram
        self.input = T.matrix('input', dtype='uint16')

        #   a sym var for input mapped to emebedded representations
        self.input_repr = T.concatenate(
            [embedding[self.input[:, i]]
             for i, embedding in enumerate(self.embeddings * self.n)],
            axis=1)

        #   activation probability of hidden variables in the positive phase
        self.hid_prb_pos = T.nnet.sigmoid(
            T.dot(self.input_repr, self.w) + self.b_hid)

        #   binary activation of hidden variables in the positive phase
        self.hid_act_pos = self.theano_rng.binomial(
            n=1, p=self.hid_prb_pos, size=self.hid_prb_pos.shape,
            dtype=theano.config.floatX)

        self.vis_prb_neg = T.nnet.sigmoid(
            T.dot(self.hid_act_pos, self.w.T) + self.b_vis)
        self.vis_act_neg = self.theano_rng.binomial(
            n=1, p=self.vis_prb_neg, size=self.vis_prb_neg.shape,
            dtype=theano.config.floatX)

        self.hid_prb_neg = T.nnet.sigmoid(
            T.dot(self.vis_act_neg, self.w) + self.b_hid)
        self.hid_act_neg = self.theano_rng.binomial(
            n=1, p=self.hid_prb_neg, size=self.hid_prb_neg.shape,
            dtype=theano.config.floatX)

        #   finally the cost function we want to reduce
        #   indirectly with contrastive divergence, and directly
        #   by optimizing the vocabulary
        self.reconstruction_error = (
            (self.input_repr - self.vis_prb_neg) ** 2).mean()

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
        vis_neg = self.vis_prb_neg
        hid_pos = self.hid_prb_pos
        hid_neg = self.hid_prb_neg
        grad_b_vis = vis_pos.mean(axis=0) - vis_neg.mean(axis=0)
        grad_b_hid = hid_pos.mean(axis=0) - hid_neg.mean(axis=0)
        grad_w = (T.dot(vis_pos.T, hid_pos) - T.dot(vis_neg.T, hid_neg)
                  ) / T.cast(vis_pos.shape[0], theano.config.floatX)

        #   calculate the "gradient" for word embedding modificaiton
        grad_l = vis_neg - vis_pos
        #   reorganize the grad_l from (N, n * d) to (N * n, d)
        grad_l = [grad_l[:, i * self.term_repr_size:
                         (i + 1) * self.term_repr_size] for i in range(self.n)]
        grad_l = T.concatenate(grad_l, axis=0)

        #   add regularization to gradients
        grad_b_vis += weight_cost * self.b_vis
        grad_b_hid += weight_cost * self.b_hid
        grad_w += weight_cost * self.w

        #   define a list of updates that happen during training
        learn_rate = T.scalar("learn_rate", dtype=theano.config.floatX)
        updates = [
            (self.w, self.w + learn_rate * alpha * grad_w),
            (self.b_vis, self.b_vis + learn_rate * alpha * grad_b_vis),
            (self.b_hid, self.b_hid + learn_rate * alpha * grad_b_hid),
        ]

        #   make a reoraganization of input from (N, n * d) into (N * n, d)
        input_stack = [self.input[:, i * len(self.embeddings):
                                  (i + 1) * len(self.embeddings)] for i in range(self.n)]
        input_stack = T.concatenate(input_stack, axis=0)

        for i, (emb, rng) in enumerate(zip(self.embeddings, self.repr_ranges)):
            updates.append((emb, T.inc_subtensor(emb[input_stack[:, i]],
                                                 learn_rate * (1 - alpha) * grad_l[:, rng[0]:rng[1]])))

        #   finally construct the function that updates parameters
        index = T.iscalar()
        train_f = theano.function(
            [index, learn_rate],
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
        for epoch_ind, epoch in enumerate(range(epochs)):
            epoch_t0 = time()

            #   calc epsilon for this epoch
            if not isinstance(eps, float):
                epoch_eps = eps(epoch_ind, train_costs_ep)
            else:
                epoch_eps = eps

            #   the number of Gibbs sampling steps
            if isinstance(steps, int):
                n_steps = steps
            else:
                n_steps = steps(epoch, train_costs_ep)

            #   iterate through the minibatches
            for batch_ind in xrange(mnb_count):
                train_costs_mnb.append(train_f(batch_ind, epoch_eps))
                log.debug('Batch train cost %.5f', train_costs_mnb[-1])

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
