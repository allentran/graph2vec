__author__ = 'porky-chu'

from collections import OrderedDict

import theano
import numpy as np
from theano import tensor as TT


class NodeVectorModel(object):
    def __init__(self, ne, de, y_max=100, alpha=0.75, seed=1692):
        """
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        """
        # parameters of the model
        np.random.seed(seed)
        self.Win = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (ne, de)).astype(theano.config.floatX))
        self.Wout = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (ne, de)).astype(theano.config.floatX))

        self.alpha = alpha

        # bundle
        self.params = [self.Win, self.Wout]
        self.names = ['Win', 'Wout']

        idxs = TT.imatrix()
        xIn = self.Win[idxs[:, 0], :]
        xOut = self.Wout[idxs[:, 1], :]

        y = TT.vector('y')  # label
        y_predictions = TT.sum(xIn * xOut, axis=1)

        fy = (y / y_max) ** alpha

        # cost and gradients and learning rate
        learning_rate = TT.scalar('lr')
        loss = TT.mean(fy * TT.square(y_predictions - TT.log(1 + y)))
        gradients = TT.grad(loss, [xIn, xOut])

        updates = [
            (self.Win, TT.inc_subtensor(self.Win[idxs[:, 0]], -learning_rate*gradients[0])),
            (self.Wout, TT.inc_subtensor(self.Wout[idxs[:, 1]], -learning_rate*gradients[1])),
        ]

        # theano functions
        self.calculate_cost = theano.function(inputs=[idxs, y], outputs=loss)
        self.classify = theano.function(inputs=[idxs], outputs=y_predictions)
        self.train = theano.function(
            inputs=[idxs, y, learning_rate],
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
