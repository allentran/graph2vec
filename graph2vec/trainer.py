__author__ = 'porky-chu'

import random
import os

import numpy as np

from node_vectors import NodeVectorModel
import parser


class Node2Vec(object):
    def __init__(self, vector_dimensions, output_dir, graph_path=None):

        self.output_dir = output_dir

        self.model = None
        self.from_nodes = None
        self.to_nodes = None
        self.dimensions = vector_dimensions
        self.from_to_idxs = None
        self.inverse_degrees = None

    def parse_graph(self, graph_path, data_dir='data', load_mmap=False):
        graph = parser.Graph(graph_path)
        self.from_nodes, self.to_nodes = graph.get_mappings()
        graph.save_mappings(self.output_dir)

        if load_mmap:
            self.from_to_idxs = np.memmap(os.path.join(data_dir, 'from_to.mat'), mode='r')
            self.inverse_degrees = np.memmap(os.path.join(data_dir, 'inverse_degrees.mat'), mode='r')
        else:
            from_to_idxs, inverse_degrees = graph.extend_graph(max_degree=2)
            self.from_to_idxs = np.memmap(os.path.join(data_dir, 'from_to.mat'), mode='w+', shape=from_to_idxs.shape)
            self.from_to_idxs[:] = from_to_idxs[:]
            self.inverse_degrees = np.memmap(os.path.join(data_dir, 'inverse_degrees.mat'), mode='w+', shape=inverse_degrees.shape)
            self.inverse_degrees[:] = inverse_degrees[:]


    def fit(self, max_epochs=100, batch_size=1000, seed=1692, cost_tol=-np.inf):
        self.model = NodeVectorModel(
            n_from=len(self.from_nodes),
            n_to=len(self.to_nodes),
            de=self.dimensions
        )

        cost0 = np.inf
        random.seed(seed)
        shuffled_idxes = np.arange(self.from_to_idxs.shape[0])
        for epoch_idx in xrange(max_epochs):

            random.shuffle(shuffled_idxes)

            cost1 = []
            for obs_idx in xrange(0, len(self.inverse_degrees), batch_size):
                cost1.append(self.model.train(self.from_to_idxs[shuffled_idxes[obs_idx:obs_idx + batch_size]],
                                          self.inverse_degrees[shuffled_idxes[obs_idx:obs_idx + batch_size]]))
                self.model.update_params(self.from_to_idxs[shuffled_idxes[obs_idx:obs_idx + batch_size]],
                                         self.inverse_degrees[shuffled_idxes[obs_idx:obs_idx + batch_size]])
                # self.model.normalize(self.from_to_idxs[shuffled_idxes[obs_idx:obs_idx + batch_size]])

            # print self.model.calculate_loss(self.from_to_idxs, self.inverse_degrees)

            cost1 = np.mean(cost1)
            if np.abs(cost0 - cost1) < cost_tol:
                return
            cost0 = cost1
            print cost1 ** 0.5


if __name__ == "__main__":
    node2vec = Node2Vec(graph_path="data/edge.list", vector_dimensions=128, output_dir='data')
    node2vec.parse_graph('data/edge.list', load_mmap=False)
    node2vec.fit(cost_tol=1e-10)
    node2vec.model.save_to_file("data/case_embeddings.pkl")
