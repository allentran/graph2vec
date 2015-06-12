__author__ = 'porky-chu'

from collections import OrderedDict

import theano
import numpy as np
from theano import tensor as TT


class NodeVectorModel(object):
    def __init__(self, ne, de, y_max=100, alpha=0.75):
        """
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        """
        # parameters of the model
        self.Win = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (ne, de)).astype(theano.config.floatX))
        self.Wout = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (ne, de)).astype(theano.config.floatX))

        # bundle
        self.params = [self.Win, self.Wout]
        self.names = ['Win', 'Wout']

        idxs = TT.imatrix()
        xIn = self.Win[idxs[:, 0], :]
        xOut = self.Wout[idxs[:, 1], :]
        n_pairs = xIn.shape[0]

        y = TT.vector('y')  # label
        y_predictions, _ = theano.scan(
            fn=lambda idx, x_in, x_out: x_in[idx, :].dot(x_out.T),
            sequences=TT.arange(n_pairs),
            outputs_info=None,
            non_sequences=[xIn, xOut],
        )

        fy = (y / y_max) ** alpha

        # cost and gradients and learning rate
        learning_rate = TT.scalar('lr')
        negative_loss = -TT.mean(fy * TT.square(y_predictions - TT.log(y)))
        gradients = TT.grad(negative_loss, self.params)
        updates = OrderedDict((p, p - learning_rate * g) for p, g in zip(self.params, gradients))

        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_predictions)
        self._train = theano.function(
            inputs=[idxs, y, learning_rate],
            outputs=negative_loss,
            updates=updates
        )

        self.normalize = theano.function(
            inputs=[],
            updates=[
                (self.Win, self.Win / TT.sqrt((self.Win ** 2).sum(axis=1)).dimshuffle(0, 'x')),
                (self.Wout, self.Wout / TT.sqrt((self.Wout ** 2).sum(axis=1)).dimshuffle(0, 'x')),
                ]
        )