"""
Implementation of a Restricted Boltzmann Machine
for language modeling. Based on (Mnih 2008.)

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

    def __init__(self, n, vocab_size, repr_size, n_hid, rng=None):

        log.info("Creating an LRBM, n=%d, vocab_size=%r, repr_size=%r"
                 ", n_hid=%d", n, vocab_size, repr_size, n_hid)

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
        self.b_repr = theano.shared(
            value=np.zeros(_n_vis, dtype=theano.config.floatX),
            name='b_repr',
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
        self.vis_neg_cond = T.dot(self.hid_act_pos, self.w[:self.repr_size].T) \
            + self.b_repr[:self.repr_size]
        #   but we sometimes need the whole visible vector, for the updates
        self.vis_neg = T.concatenate(
            (self.vis_neg_cond, input_repr[:, self.repr_size:]), axis=1)

        #   a function that returns the energy symbolic variable
        #   given visible and hidden unit symbolic variables
        def energy(visible, hidden):
            return -T.dot(visible, self.b_repr.T) \
                - T.dot(hidden, self.b_hid.T) \
                - (T.dot(visible, self.w) * hidden).sum(axis=1)

        #   activation probability, hidden layer, negative phase
        self.hid_prb_neg = T.nnet.sigmoid(
            T.dot(self.vis_neg, self.w) + self.b_hid)

        #   standard energy of the input
        self.energy = energy(input_repr, self.hid_prb_pos)

        #   exact probablity of a single sample, given model params only
        #   first define a function that calculates the prob. of one sample
        batch_size = self.input.shape[0]
        _partition = T.concatenate([
            emb.dimshuffle('x', 0, 1).repeat(batch_size, 0),
            emb[self.input[:, 1:]].reshape((batch_size, 1, -1)).repeat(self.vocab_size, 1)
        ], axis=2)
        #   input to each hidden unit, for every input-repr in _partition
        _hid_in = T.dot(_partition, self.w) + self.b_hid
        #   a binom signifying a hidden unit being on or off
        _hid_in_exp = (1 + T.exp(_hid_in))
        #   divide with mean for greater numeric stability
        #   does not change end probability
        _hid_in_exp /= _hid_in_exp.mean(axis=2).mean(axis=1).dimshuffle(0, 'x', 'x')
        _probs = _hid_in_exp.prod(axis=2)
        _probs *= T.exp(T.dot(_partition, self.b_repr))
        #   to enable the usage of smoothing, we'll expose the unnormalized partition
        self.distr_w_unn = _probs
        self.distr_w = _probs / _probs.sum(axis=1).dimshuffle(0, 'x')

        #   we also need the probability of the conditioned term
        self.probability = self.distr_w[T.arange(input.shape[0]), input[:, 0]]

        #   returns the unnormalized probabilities of a set of samples
        #   useful only for relative comparison of samples
        _unnp_hid_in = T.dot(self.input_repr, self.w) + self.b_hid
        _unnp_hid_in_exp = (1 + T.exp(_unnp_hid_in))
        _unnp_hid_in_exp /= _unnp_hid_in_exp.mean()
        _unnp_probs = _unnp_hid_in_exp.prod(axis=1)
        _unnp_probs *= T.exp(T.dot(self.input_repr, self.b_repr))
        self.unnp = _unnp_probs

        #   distribution of the vocabulary, given hidden state
        #   we use vis_neg because it's exactly W * hid_act
        _partition_given_h = -T.dot(self.vis_neg_cond, emb.T)
        _partition_given_h -= _partition_given_h.min(axis=1).dimshuffle(0, 'x')
        _partition_given_h = T.exp(-_partition_given_h)
        self.distribution_w_given_h = _partition_given_h / _partition_given_h.sum(axis=1).dimshuffle(0, 'x')

        #   reconstruction error that we want to reduce
        #   we use contrastive divergence to model the distribution
        #   and optimize the vocabulary
        self.cost = (
            (input_repr - self.vis_neg) ** 2).mean()

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
            "b_hid": self.b_hid,
            "b_repr": self.b_repr,
        }

        if not symbolic:
            r_val = dict([(k, v.get_value(borrow=True))
                          for k, v in r_val.iteritems()])

        return r_val

    def train(self, x_train, x_valid, mnb_size, epochs, eps, alpha,
              steps=1, weight_cost=1e-4):
        """
        Trains the LRBM with the given data. Returns a tuple containing
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
            (LRBM) learning is multiplied with alpha, while representation
            learning (word-vectors) is multiplied with (1 - alpha).
        :param steps: The number of steps to be used in PCD.
            Integer or callable, or a callable that determines the
            learning rate based on epoch number and a list of
            error rates.
        :param weight_cost: Regularization cost for L2 regularization
            (weight decay).
        """

        log.info('Training LRBM, epochs: %d, eps: %r, alpha: %.2f, steps:%d',
                 epochs, eps, alpha, steps)

        #   pack trainset into a shared variable
        mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
        x_train = theano.shared(x_train, name='x_train', borrow=True)

        #   *** Creating a function for training the net

        #   first calculate CD "gradients"
        vis_pos = self.input_repr
        vis_neg = self.vis_neg
        hid_pos = self.hid_prb_pos
        hid_neg = self.hid_prb_neg
        grad_b_repr = vis_pos.mean(axis=0) - vis_neg.mean(axis=0)
        grad_b_hid = hid_pos.mean(axis=0) - hid_neg.mean(axis=0)
        grad_w = (T.dot(vis_pos.T, hid_pos) - T.dot(vis_neg.T, hid_neg)
                  ) / T.cast(vis_pos.shape[0], theano.config.floatX)

        #   calculate the "gradient" for word embedding modification
        #   first define a function that calcs it for one sample
        def _grad_l_for_sample(w, p_w_given_h, h_v_pos, h_v_neg):
            #   reshape from (n * d, ) to (n, d)
            h_v_pos = T.reshape(h_v_pos, (self.n, self.repr_size))
            h_v_neg = T.reshape(h_v_neg, (self.n, self.repr_size))
            #   tile p from (Nw, 1) to (Nw, n)
            p_w_given_h = T.tile(p_w_given_h.dimshuffle((0, 'x')),
                                 (1, self.n))
            #   first the negative phase gradient
            #   to form a matrix of appropriate size
            grad_l = T.dot(p_w_given_h, -h_v_neg)
            #   now the positive phase
            grad_l = T.inc_subtensor(grad_l[w], h_v_pos)
            return grad_l
        #   now calculate it for all the samples
        _grad_l, _ = theano.scan(
            _grad_l_for_sample,
            sequences=[self.input, self.distribution_w_given_h,
                       T.dot(hid_pos, self.w.T), T.dot(hid_neg, self.w.T)])
        #   final gradient is just the mean across minibatch samples
        grad_l = _grad_l.mean(axis=0)

        #   add regularization to gradients
        grad_w -= weight_cost * self.w

        #   define a list of updates that happen during training
        eps_th = T.scalar("eps", dtype=theano.config.floatX)
        updates = [
            (self.w, self.w + eps_th * alpha * grad_w),
            (self.b_repr, self.b_repr + eps_th * alpha * grad_b_repr),
            (self.b_hid, self.b_hid + eps_th * alpha * grad_b_hid),
            (self.embedding, self.embedding + eps_th * (1 - alpha) * grad_l)
        ]

        #   finally construct the function that updates parameters
        index = T.iscalar()
        train_f = theano.function(
            [index, eps_th],
            self.cost,
            updates=updates,
            givens={
                self.input: x_train[index * mnb_size: (index + 1) * mnb_size]
            }
        )
        #   ***  Done creating a function for training the net ###

        #   a separate function we will use for validation
        validate_f = theano.function(
            [self.input],
            self.cost,
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

            #   calc epsilon for this epoch
            if not isinstance(eps, float):
                epoch_eps = eps(epoch_ind, train_costs)
            else:
                epoch_eps = eps

            #   iterate learning through the minibatches
            def mnb_train(batch_ind):
                cost = train_f(batch_ind, epoch_eps)
                log.debug('Mnb %d train cost %.5f', batch_ind, cost)
                mnb_callback(self, epoch_ind, batch_ind)
                return cost
            train_costs.append(np.mean(map(mnb_train, xrange(mnb_count))))

            valid_costs.append(validate_f(x_valid))
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
