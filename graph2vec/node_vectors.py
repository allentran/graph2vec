__author__ = 'porky-chu'

import cPickle

import theano
import numpy as np
from theano import tensor as TT


class NodeVectorModel(object):
    def __init__(self, n_from, n_to, de, seed=1692, init_params=None):
        """
        n_from :: number of from embeddings in the vocabulary
        n_to :: number of to embeddings in the vocabulary
        de :: dimension of the word embeddings
        """
        np.random.seed(seed)
        # parameters of the model
        if init_params is not None:
            with open('data/case_embeddings.pkl', 'rb') as f:
                temp = cPickle.load(f)
            self.Win = theano.shared(temp.Win.get_value().astype(theano.config.floatX))
            self.Wout = theano.shared(temp.Wout.get_value().astype(theano.config.floatX))
        else:
            self.Win = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (n_from, de)).astype(theano.config.floatX))
            self.Wout = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (n_to, de)).astype(theano.config.floatX))

        # adagrad
        self.cumulative_gradients_in = theano.shared(0.1 * np.ones((n_from, de)).astype(theano.config.floatX))
        self.cumulative_gradients_out = theano.shared(0.1 * np.ones((n_to, de)).astype(theano.config.floatX))

        idxs = TT.imatrix()
        x_in = self.Win[idxs[:, 0], :]
        x_out = self.Wout[idxs[:, 1], :]

        norms_in= TT.sqrt(TT.sum(x_in ** 2, axis=1))
        norms_out = TT.sqrt(TT.sum(x_out ** 2, axis=1))
        norms = norms_in * norms_out

        y = TT.vector('y')  # label
        y_predictions = TT.sum(x_in * x_out, axis=1) / norms

        # cost and gradients and learning rate
        loss = TT.mean(TT.sqr(y_predictions - y))
        gradients = TT.grad(loss, [x_in, x_out])

        updates = [
            (self.cumulative_gradients_in, TT.inc_subtensor(self.cumulative_gradients_in[idxs[:, 0]], gradients[0] ** 2)),
            (self.cumulative_gradients_out, TT.inc_subtensor(self.cumulative_gradients_out[idxs[:, 1]], gradients[1] ** 2)),
            (self.Win, TT.inc_subtensor(self.Win[idxs[:, 0]], - (0.5 / TT.sqrt(self.cumulative_gradients_in[idxs[:, 0]])) * gradients[0])),
            (self.Wout, TT.inc_subtensor(self.Wout[idxs[:, 1]], - (0.5 / TT.sqrt(self.cumulative_gradients_out[idxs[:, 1]])) * gradients[1])),
        ]

        # theano functions
        self.calculate_loss = theano.function(inputs=[idxs, y], outputs=loss)
        self.classify = theano.function(inputs=[idxs], outputs=y_predictions)
        self.train = theano.function(
            inputs=[idxs, y],
            outputs=loss,
            updates=updates,
            name='training_fn'
        )

    def __getstate__(self):
        return self.Win, self.Wout

    def __setstate__(self, state):
        Win, Wout = state
        self.Win = Win
        self.Wout = Wout

    def save_to_file(self, output_path):
        with open(output_path, 'wb') as output_file:
            cPickle.dump(self, output_file, cPickle.HIGHEST_PROTOCOL)