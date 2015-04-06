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

        #   input is a matrix of (N, len(repr_sizes)) dimensionality
        self.input = T.matrix('input', dtype='int32')

        #   random number generators
        if rng is None:
            numpy_rng = np.random.RandomState()
        elif isinstance(rng, int):
            numpy_rng = np.random.RandomState(rng)
        else:
            numpy_rng = rng

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

        #   create word embeddings
        self.embeddings = map(create_dim_var, vocab_sizes, repr_sizes)

        #   figure out how many visible variables there are
        _n_vis = n * sum(repr_sizes)

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
        self.init_vars

    def init_vars(self):

        #   a sym var for input mapped to emebedded representations
        self.input_repr = T.concatenate(
            [embedding[self.input[:, i]]
             for i, embedding in enumerate(self.embeddings)],
            axis=1)

        #   activation probability of hidden variables in the positive phase
        self.hid_prb_pos = T.nnet.sigmoid(
            T.dot(self.input_repr, self.w) + self.b_hid)

        #   binary activation of hidden variables in the positive phase
        self.hid_act_pos = self.theano_rng.binomial(
            n=1, p=self.hid_prb, size=self.hid_prb.shape,
            dtype=theano.config.floatX)

        #   contrastive divergence steps
        #   let's start with defining the function of a single step
        def cd_step(step_hid_input):
            #   calculate visible layer probabilities and activations
            step_vis_prb = T.nnet.sigmoid(
                T.dot(step_hid_input, self.w.T) + self.b_vis)
            step_vis_act = self.theano_rng.binomial(
                n=1, p=step_vis_prb, size=step_vis_prb.shape,
                dtype=theano.config.floatX)

            #   calculate hidden layer probabilities and activations
            step_hid_prb = T.nnet.sigmoid(
                T.dot(step_vis_act, self.w) + self.b_hid)
            step_hid_act = self.theano_rng.binomial(
                n=1, p=step_hid_prb, size=step_hid_prb.shape,
                dtype=theano.config.floatX)

            return (step_vis_prb, step_vis_act, step_hid_prb, step_hid_act)

        #   then we prepare the scan function
        #   symbolic variable for the number of contrastive divergences steps
        self.cd_steps = T.scalar('n_steps', dtype='int8')
        scan_res, _ = theano.scan(
            cd_step,
            outputs_info=[None, None, None, self.hid_act_pos],
            n_steps=self.cd_steps)

        #   scan returns probabilities and activations for al CD steps
        #   we only need the probabilities for the last step
        self.vis_prb_neg = scan_res[0][-1]
        self.hid_prb_neg = scan_res[2][-1]

        #   finally the cost function we want to reduce
        #   indirectly with contrastive divergence, and directly
        #   by optimizing the vocabulary
        self.reconstruction_error = (
            (self.input_repr - self.vis_prb_neg) ** 2).mean()

    def train(self, x_mnb, epochs, eps,
              steps=1, spars=None, spars_cost=None,
              weight_cost=1e-4):
        """
        Trains the RBM with the given data. Returns a tuple containing
        (costs, times, hid_unit_activation_histograms). All three
        elements are lists, one item per epoch except for 'times' that
        has an extra element (training start time).

        :param x_mnb: Trainset split into minibatches. Thus,
            x_mnb is an iterable containing numpy arrays of
            (mnb_N, n_vis) shape, where mnb_N is the number of
            samples in the minibatch.

        :param epochs: Number of epochs (int) of training.

        :param eps: Learning rate. Either a float value to be
            used directly, or a callable that determines the
            learning rate based on epoch number and a list of
            error rates.

        :param steps: The number of steps to be used in PCD / CD.
            Integer or callable, or a callable that determines the
            learning rate based on epoch number and a list of
            error rates.

        :param spars: Target sparsity of hidden unit activation.

        :param spars_cost: Cost of deviation from the target sparsity.

        :param weight_cost: Regularization cost for L2 regularization
            (weight decay).

        """

        log.info('Training RBM, epochs: %d, eps: %r, steps:%d, '
                 'spars:%r, spars_cost:%r',
                 epochs, eps, steps, spars, spars_cost)

        #   things we'll track through training, for reporting
        epoch_costs = []
        epoch_times = []
        epoch_hid_prbs = np.zeros((epochs, self.n_hid))

        #   iterate through the epochs
        for epoch_ind, epoch in enumerate(range(epochs)):
            log.info('Starting epoch %d', epoch)
            epoch_t0 = time()

            #   calc epsilon for this epoch
            if not isinstance(eps, float):
                epoch_eps = eps(epoch_ind, epoch_costs)
            else:
                epoch_eps = eps

            #   iterate through the minibatches
            batch_costs = []
            for batch_ind, batch in enumerate(x_mnb):

                #   positive statistics
                #   _prb suffix indicates activation probabilities
                #   _act suffix indicates binary activation
                pos_hid_prb, pos_hid_act = self.hid_given_vis(batch)

                #   the number of Gibbs sampling steps
                if isinstance(steps, int):
                    n_steps = steps
                else:
                    n_steps = steps(epoch, epoch_costs)

                #   do Gibbs sampling
                neg_vis_prb, _, neg_hid_prb, neg_hid_act = \
                    self.steps_given_hid(pos_hid_act, n_steps)
                #   the scan function returns all steps
                #   we don't need them all
                neg_hid_act = neg_hid_act[-1]
                neg_vis_prb = neg_vis_prb[-1]
                neg_hid_prb = neg_hid_prb[-1]

                #   gradients based on pos/neg statistics
                pos_vis = batch.mean(axis=0, dtype=theano.config.floatX)
                pos_hid = pos_hid_prb.mean(axis=0)
                grad_b_vis = pos_vis - neg_vis_prb.mean(axis=0)
                grad_b_hid = pos_hid - neg_hid_prb.mean(axis=0)
                grad_w = (np.dot(batch.T, pos_hid_prb) - np.dot(
                    neg_vis_prb.T, neg_hid_prb)) / len(batch)

                #   L2 regularization gradient
                # grad_b_vis += weight_cost * self.b_vis.get_value()
                grad_b_hid += weight_cost * self.b_hid.get_value()
                grad_w += weight_cost * self.w.get_value()

                #   sparsity gradient
                if((spars is not None) & (spars_cost is not None)
                        & (spars_cost != 0.0)):
                    spars_grad = (pos_hid - spars) * spars_cost
                    grad_w -= np.dot(pos_vis.reshape((self.n_vis, 1)),
                                     spars_grad.reshape((1, self.n_hid)))
                    grad_b_hid -= spars_grad

                #   calc cost to be reported
                batch_costs.append(((neg_vis_prb - batch) ** 2).mean())

                #   hidden unit activation probability reporting
                #   note that batch.shape[0] is the number of samples in batch
                epoch_hid_prbs[epoch_ind, :] += pos_hid / batch.shape[0]

                #   updating the params
                self.w.set_value(self.w.get_value() + epoch_eps * grad_w)
                self.b_vis.set_value(
                    self.b_vis.get_value() + epoch_eps * grad_b_vis)
                self.b_hid.set_value(
                    self.b_hid.get_value() + epoch_eps * grad_b_hid)

            epoch_costs.append(np.array(batch_costs).mean())
            epoch_times.append(time() - epoch_t0)
            log.info(
                'Epoch cost %.5f, duration %.2f sec',
                epoch_costs[-1],
                epoch_times[-1]
            )

        log.info('Training duration %.2f min',
                 (sum(epoch_times)) / 60.0)

        return epoch_costs, epoch_times, epoch_hid_prbs

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
