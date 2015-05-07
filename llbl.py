"""
Implementation of a log-bilinear language model,
based on (Mnih, 2008.).
"""
import numpy as np
import theano
import theano.tensor as T
import logging
from time import time

log = logging.getLogger(__name__)


class LLBL():

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
        def make_w(i):
            return theano.shared(value=np.asarray(
                numpy_rng.uniform(
                    low=-0.01, high=0.01, size=(repr_size, repr_size)),
                dtype=theano.config.floatX
            ), name='w_%d' % i, borrow=True)
        self.w = map(make_w, range(n - 1))

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

        #   energy function
        def energy(input):
            """
            Returns the symbolic variable for the energy, given
            an input symbolic variable.

            :param input: Symbolic variable for ngrams. Shaped
                (N, n).
            """
            cond_term_repr = self.embedding[input[:, 0]]
            energy = -self.b_word[input[:, 0]]
            energy -= T.dot(cond_term_repr, self.b_repr)
            for term in xrange(self.n - 1):
                term_proj = T.dot(emb[input[:, term + 1]], self.w[term])
                energy -= (term_proj * cond_term_repr).sum(axis=1)
            energy.name = 'energy'
            return energy

        self.energy = energy(input)

        #   exact probablity distribution of words for context
        #   first define a function that calculates the prob. of one sample
        def _probability(sample):
            #   a matrix with all words as the conditioned term,
            #   with conditioning terms like in sample
            partition = T.concatenate([
                T.arange(self.vocab_size).dimshuffle(0, 'x'),
                T.tile(sample[1:].flatten().dimshuffle(('x', 0)),
                       (self.vocab_size, 1))],
                axis=1)

            #   energy for all the terms
            partition_en = energy(partition)
            #   subract C (equal to dividing with exp(C) in exp-space)
            #   so that exp(_partition) fits in float32
            partition_en -= partition_en.min()
            #   exponentiate, normalize and return
            partition = T.exp(-partition_en)
            return partition / partition.sum()

        #   now use theano scan to calculate probabilities for all inputs
        self.distr_w, _ = theano.scan(_probability,
                                      outputs_info=None,
                                      sequences=[input])

        #   we also need the probability of the conditioned term
        self.probability = self.distr_w[T.arange(input.shape[0]), input[:, 0]]

        #   returns the unnormalized probabilities of a set of samples
        #   useful only for relative comparison of samples
        unnp_en = -energy(input)
        unnp_en -= unnp_en.min()
        unnp = T.exp(-unnp_en)
        self.unnp = unnp / unnp.sum()

        #   cost function we want to reduce
        #   good old log loss
        self.cost = -T.log(self.probability).mean()

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
            "embedding": self.embedding,
            "b_repr": self.b_repr,
            "b_word": self.b_word,
        }
        for i, w_i in enumerate(self.w):
            r_val["w_%d" % i] = w_i

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
        mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
        x_train = theano.shared(x_train, name='x_train', borrow=True)

        #   *** Creating a function for training the net

        emb = self.embedding
        inp = self.input
        inp_sz = inp.shape[0].astype(theano.config.floatX)

        #   grad_w, per term-combination-matrix
        grad_w = []
        for i in xrange(1, self.n):
            emb_i = emb[inp[:, i]].T
            pos = T.dot(emb_i, emb[inp[:, 0]]) / inp_sz
            neg = T.dot(emb_i, T.dot(self.distr_w, self.embedding))
            grad_w.append(pos - neg)

        #   word embedding gradient
        grad_emb = T.zeros_like(emb)
        grad_emb = T.inc_subtensor(grad_emb[inp[:, 0]], self.b_repr / inp_sz)
        for term in range(self.n - 1):
            #   P(data) projection of term_i onto the conditioned term
            proj_term = T.dot(emb[inp[:, term + 1]], self.w[term]) / inp_sz
            grad_emb = T.inc_subtensor(grad_emb[inp[:, 0]], proj_term)

            #   P(data) projection of conditioned term on term_i
            proj_cond = T.dot(emb[inp[:, 0]], self.w[term].T) / inp_sz
            grad_emb = T.inc_subtensor(grad_emb[inp[:, term + 1]], proj_cond)

            #   P(model) projection of term_i onto the conditioned
            proj_term = T.dot(emb[inp[:, term + 1]], self.w[term])
            grad_emb += T.dot(self.distr_w.T, proj_term)

            #   P(model) projection of conditioned term on term_i
            proj_cond = T.dot(T.dot(self.distr_w, emb), self.w[term].T)
            grad_emb = T.inc_subtensor(grad_emb[inp[:, term + 1]], proj_cond)

        #   biases gradients
        grad_b_repr = emb[inp[:, 0]].mean(
            axis=0) - T.dot(self.distr_w, emb).mean()
        grad_b_word = - self.distr_w.mean(axis=0)
        grad_b_word = T.inc_subtensor(grad_b_word[inp[:, 0]], 1. / inp_sz)

        #   add L2 regularization to gradients
        grad_w = map(lambda g_w, w: g_w - weight_cost * w, grad_w, self.w)
        grad_emb -= weight_cost * emb

        #   define a list of updates that happen during training
        eps_th = T.scalar("eps", dtype=theano.config.floatX)
        updates = [
            (self.b_repr, self.b_repr + eps_th * alpha * grad_b_repr),
            (self.b_word, self.b_word + eps_th * alpha * grad_b_word),
            (emb, emb + eps_th * (1 - alpha) * grad_emb)
        ]
        for grad_w, w in zip(grad_w, self.w):
            updates.append((w, w + eps_th * alpha * grad_w))

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
