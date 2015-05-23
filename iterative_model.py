
import logging
from time import time
import abc

import numpy as np
import theano
import theano.tensor as T

import grad_descent
import util

log = logging.getLogger(__name__)


class IterativeModel():
    """
    Base class for language models that are trained iteratively
    using gradient descent.
    """

    @abc.abstractmethod
    def params(self, symbolic=False):
        """
        Returns a dictionary of all the model parameters. The dictionary
        keys are parameter names, values are parameter.

        :param symbolic: If symbolic Theano variables should be returned.
            If False (default), then their numpy value arrays are returned.
        :return: A dictionary that maps parameter names to their values
            (shared theano variables, or numpy arrays).
        """
        return

    def evaluate(self, x, mnb_size):
        """
        Evaluates model cost on given samples and returns
        the mean.

        :param x: Samples, a numpy array of shape (N, model_input).
        :param mnb_size: Minibatch size, necessary because evaluating
            all the samples in 'x' at once might be too memory demanding.
        :return: Mean cost of samples in 'x'.
        """

        evaluate_f = getattr(self, "_evaluate", None)
        if evaluate_f is None:
            evaluate_f = theano.function(
                [self.input],
                self.cost)
            self._evaluate = evaluate_f

        #   take into account possibly unbalance mnb sizes
        return np.sum(
            [evaluate_f(mnb) * mnb.shape[0]
             for mnb in util.create_minibatches(x, None, mnb_size, False)
             ]) / x.shape[0]

    def train(self, x_train, mnb_size, epochs, eps, weight_cost=1e-4):
        """
        Trains the LMLP with the given data. Returns a tuple containing
        (costs, times, hid_unit_activation_histograms). All three
        elements are lists, one item per epoch except for 'times' that
        has an extra element (training start time).

        :param x_train: Trainset of (N, n_vis) shape, where N is the
            number of samples.
        :param mnb_size: Minibatch size, the number of samples in the
            minibatch.
        :param epochs: Number of epochs (int) of training.
        :param eps: Learning rate. Either a float value to be
            used directly, or a callable that determines the
            learning rate based on epoch number and a list of
            error rates.
        :param weight_cost: Regularization cost for L2 regularization
            (weight decay).
        """

        log.info('Training LMLP, epochs: %d, eps: %r', epochs, eps)

        #   pack trainset into a shared variable
        train_mnb_count = (x_train.shape[0] - 1) / mnb_size + 1
        x_train = theano.shared(x_train, name='x_train', borrow=True)

        #  gradient optimization, use RMSprop
        param_updates = grad_descent.gradient_updates_rms(
            self.cost_l2, self.params(True).values(), eps, 0.9)

        #   construct the function that updates parameters
        index = T.iscalar()
        train_f = theano.function(
            [index, self.l2_lmbd],
            self.cost,
            updates=param_updates.updates,
            givens={
                self.input: x_train[index * mnb_size: (index + 1) * mnb_size]
            }
        )

        #   things we'll track through training, for reporting
        train_costs = []
        train_times = []

        #   iterate through the epochs
        log.info("Starting training")
        #   after each epoch, call a callback if provided
        epoch_callback = getattr(self, "epoch_callback", lambda a, b: True)
        mnb_callback = getattr(self, "mnb_callback", lambda a, b, c: True)
        for epoch_ind, epoch in enumerate(range(epochs)):
            epoch_t0 = time()

            #   calc epsilon for this epoch
            if not isinstance(eps, float):
                epoch_eps = eps(epoch_ind, train_costs)
            else:
                epoch_eps = eps * (0.1 + 0.9 * 0.95 ** epoch_ind)

            #   iterate learning through the minibatches
            def mnb_train(batch_ind):
                cost = train_f(batch_ind, epoch_eps)
                log.debug('Mnb %d train cost %.5f', batch_ind, cost)
                mnb_callback(self, epoch_ind, batch_ind)
                return cost
            train_costs.append(
                np.mean(map(mnb_train, xrange(train_mnb_count))))

            train_times.append(time() - epoch_t0)
            log.info('Epoch %d, duration %.2f sec, train cost: %.5f',
                     epoch_ind, train_times[-1], train_costs[-1])

            if not epoch_callback(self, epoch_ind):
                break

        log.info('Training duration %.2f min', (sum(train_times)) / 60.0)

        return train_costs

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
