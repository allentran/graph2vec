__author__ = 'porky-chu'

import numpy as np
import random

from node_vectors import NodeVectorModel


class Node2Vec(object):

    def __init__(self, graph_path, vector_dimensions):

        self.n = None
        self.data = None
        self.from_to_idxs = None
        self.model = None
        self.dimensions = vector_dimensions

        self.load_graph(graph_path)

    def load_graph(self, graph_path):

        self.from_to_idxs = []
        self.data = []
        max_index = 0

        with open(graph_path, 'r') as graph_file:
            for line in graph_file:
                parsed_line = line.strip().split(' ')
                if len(parsed_line) in [2, 3]:
                    first = int(parsed_line[0])
                    second = int(parsed_line[1])
                    max_index_in_line = np.max([first, second])
                    if max_index_in_line > max_index:
                        max_index = max_index_in_line
                    self.from_to_idxs.append([first, second])
                    if len(parsed_line) == 3:
                        self.data.append(int(parsed_line[2]))
                    else:
                        self.data.append(1)

        self.n = max_index + 1
        self.data = np.array(self.data).astype(np.float32)
        self.from_to_idxs = np.array(self.from_to_idxs).astype(np.int32)

    def fit(self, max_epochs=100, batch_size=100000, seed=1692):
        self.model = NodeVectorModel(
            ne=self.n,
            de=self.dimensions
        )

        for epoch_idx in xrange(max_epochs):

            random.seed(seed)
            random.shuffle(self.from_to_idxs)
            random.seed(seed)
            random.shuffle(self.data)

            seed = random.randint(0, 1e5)

            for obs_idx in xrange(0, self.n, batch_size):
                self.model.train(self.from_to_idxs[obs_idx:obs_idx+batch_size], self.data[obs_idx:obs_idx+batch_size], 0.05)
                self.model.normalize(self.from_to_idxs[obs_idx:obs_idx+batch_size])
            print self.model.calculate_cost(self.from_to_idxs, self.data)

if __name__ == "__main__":
    node2vec = Node2Vec(graph_path="data/edge.list", vector_dimensions=100)
    node2vec.fit()
