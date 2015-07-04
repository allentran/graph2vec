__author__ = 'porky-chu'

import random
import os
import logging

import numpy as np

from node_vectors import NodeVectorModel
import parser


class Graph2Vec(object):
    def __init__(self, vector_dimensions, output_dir='data'):

        self.output_dir = output_dir

        self.model = None
        self.from_nodes = None
        self.to_nodes = None
        self.dimensions = vector_dimensions
        self.from_to_idxs = None
        self.inverse_degrees = None

    def parse_graph(self, graph_path, data_dir='data', load_edges=False, extend_paths=2):
        graph = parser.Graph(graph_path)
        self.from_nodes, self.to_nodes = graph.get_mappings()
        graph.save_mappings(self.output_dir)

        if load_edges:
            self.inverse_degrees = np.memmap(
                os.path.join(data_dir, 'inverse_degrees.mat'),
                mode='r',
                dtype='float32'
            )
            self.from_to_idxs = np.memmap(
                os.path.join(data_dir, 'from_to.mat'),
                mode='r',
                dtype='int32'
            )
            self.from_to_idxs = np.reshape(self.from_to_idxs, newshape=(self.inverse_degrees.shape[0], 2))
        else:
            from_to_idxs, inverse_degrees = graph.extend_graph(max_degree=extend_paths)
            self.from_to_idxs = np.memmap(
                os.path.join(data_dir, 'from_to.mat'),
                mode='r+',
                shape=from_to_idxs.shape,
                dtype='int32'
            )
            self.from_to_idxs[:] = from_to_idxs[:]
            self.inverse_degrees = np.memmap(
                os.path.join(data_dir, 'inverse_degrees.mat'),
                mode='r+',
                shape=inverse_degrees.shape,
                dtype='float32'
            )
            self.inverse_degrees[:] = inverse_degrees[:]


    def fit(self, max_epochs=100, batch_size=1000, seed=1692, params=None):

        self.model = NodeVectorModel(
            n_from=len(self.from_nodes),
            n_to=len(self.to_nodes),
            de=self.dimensions,
            init_params=params,
        )

        random.seed(seed)
        shuffled_idxes = np.arange(self.from_to_idxs.shape[0])
        for epoch_idx in xrange(max_epochs):

            random.shuffle(shuffled_idxes)

            cost = []
            for obs_idx in xrange(0, len(self.inverse_degrees), batch_size):
                cost.append(self.model.train(self.from_to_idxs[shuffled_idxes[obs_idx:obs_idx + batch_size]],
                                          self.inverse_degrees[shuffled_idxes[obs_idx:obs_idx + batch_size]]))

            cost = np.mean(cost)
            logging.info('After %s epochs, cost=%s' % (epoch_idx, cost ** 0.5))

def main():
    node2vec = Graph2Vec(vector_dimensions=128)
    node2vec.parse_graph('data/edge.list', load_edges=True)
    node2vec.fit("data/case_embeddings.pkl")
    node2vec.model.save_to_file("data/case_embeddings.pkl")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
