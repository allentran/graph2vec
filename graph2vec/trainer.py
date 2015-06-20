__author__ = 'porky-chu'

import random

import numpy as np

from node_vectors import NodeVectorModel
import parser


class Node2Vec(object):
    def __init__(self, graph_path, vector_dimensions, output_dir):

        self.output_dir = output_dir

        self.model = None
        self.from_nodes = None
        self.to_nodes = None
        self.dimensions = vector_dimensions
        self.from_to_idxs = None
        self.inverse_degrees = None

        self.parse_graph(graph_path)

    def parse_graph(self, graph_path):
        graph = parser.Graph(graph_path)
        self.from_nodes, self.to_nodes = graph.get_mappings()
        graph.save_mappings(self.output_dir)
        self.from_to_idxs, self.inverse_degrees = graph.extend_graph(max_degree=2)


    def fit(self, max_epochs=100, batch_size=100, seed=1692, cost_tol=-np.inf):

        self.model = NodeVectorModel(
            n_from=len(self.from_nodes),
            n_to=len(self.to_nodes),
            de=self.dimensions
        )

        cost0 = np.inf
        random.seed(seed)
        for epoch_idx in xrange(max_epochs):

            random.seed(seed)
            random.shuffle(self.from_to_idxs)
            random.seed(seed)
            random.shuffle(self.inverse_degrees)

            seed = random.randint(0, 1e5)
            cost1 = 0
            for obs_idx in xrange(0, len(self.inverse_degrees), batch_size):
                cost1 += self.model.train(self.from_to_idxs[obs_idx:obs_idx + batch_size],
                                          self.inverse_degrees[obs_idx:obs_idx + batch_size])
                self.model.update_params(self.from_to_idxs[obs_idx:obs_idx + batch_size],
                                         self.inverse_degrees[obs_idx:obs_idx + batch_size])
                self.model.normalize(self.from_to_idxs[obs_idx:obs_idx + batch_size])

            cost1 /= len(self.inverse_degrees / batch_size)
            if np.abs(cost0 - cost1) < cost_tol:
                return
            cost0 = cost1
            print cost1


if __name__ == "__main__":
    node2vec = Node2Vec(graph_path="data/edge.list", vector_dimensions=50, output_dir='data')
    node2vec.fit(cost_tol=1e-10)
    node2vec.model.save_to_file("data/case_embeddings.pkl")
