__author__ = 'porky-chu'

import cPickle

import theano
import numpy as np
from theano import tensor as TT


class NodeVectorModel(object):
    def __init__(self, n_from, n_to, de, seed=1692):
        """
        n_from :: number of from embeddings in the vocabulary
        n_to :: number of to embeddings in the vocabulary
        de :: dimension of the word embeddings
        """
        # parameters of the model
        np.random.seed(seed)
        self.Win = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (n_from, de)).astype(theano.config.floatX))
        self.Wout = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (n_to, de)).astype(theano.config.floatX))

        # adagrad
        self.cumulative_gradients_in = theano.shared(np.zeros((n_from, de)).astype(theano.config.floatX))
        self.cumulative_gradients_out = theano.shared(np.zeros((n_to, de)).astype(theano.config.floatX))

        idxs = TT.imatrix()
        xIn = self.Win[idxs[:, 0], :]
        xOut = self.Wout[idxs[:, 1], :]

        x_in_norm = TT.sqrt((xIn ** 2).sum(axis=1))
        x_out_norm = TT.sqrt((xOut ** 2).sum(axis=1))

        y = TT.vector('y')  # label
        y_predictions = TT.sum(xIn * xOut, axis=1) / (x_in_norm * x_out_norm)

        # cost and gradients and learning rate
        loss = TT.mean(TT.sqr(y_predictions - TT.log(1 + y)))
        gradients = TT.grad(loss, [xIn, xOut])

        updates = [
            (self.cumulative_gradients_in, TT.inc_subtensor(self.cumulative_gradients_in[idxs[:, 0]], gradients[0] ** 2)),
            (self.cumulative_gradients_out, TT.inc_subtensor(self.cumulative_gradients_out[idxs[:, 1]], gradients[1] ** 2)),
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

        self.normalize = theano.function(
            inputs=[idxs],
            updates=[
                (
                    self.Win,
                    TT.set_subtensor(self.Win[idxs[:, 0]], self.Win[idxs[:, 0]] / TT.sqrt((self.Win[idxs[:, 0]] ** 2).sum(axis=1)).dimshuffle(0, 'x'))
                ),
                (
                    self.Wout,
                    TT.set_subtensor(self.Wout[idxs[:, 1]], self.Wout[idxs[:, 1]] / TT.sqrt((self.Wout[idxs[:, 1]] ** 2).sum(axis=1)).dimshuffle(0, 'x'))
                )
                ]
        )

        self.update_params = theano.function(
            inputs=[idxs, y],
            updates=[
                (self.Win, TT.inc_subtensor(self.Win[idxs[:, 0]], - (0.5 / TT.sqrt(self.cumulative_gradients_in[idxs[:, 0]])) * gradients[0])),
                (self.Wout, TT.inc_subtensor(self.Wout[idxs[:, 1]], - (0.5 / TT.sqrt(self.cumulative_gradients_out[idxs[:, 1]])) * gradients[1])),
            ]
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