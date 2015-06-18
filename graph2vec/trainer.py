__author__ = 'porky-chu'

import random
import json
import os

import numpy as np

from node_vectors import NodeVectorModel


class Node2Vec(object):
    def __init__(self, graph_path, vector_dimensions):

        self.n = 0

        self.from_dict = {None: -1}
        self.to_dict = {None: -1}

        self.data = None
        self.from_to_idxs = None
        self.model = None
        self.dimensions = vector_dimensions

        self.load_graph(graph_path)
        self.from_dict.pop(None)
        self.to_dict.pop(None)

    def _update_get_index(self, key, node_dict):

        if key not in node_dict:
            node_dict[key] = len(node_dict) + 1
        return node_dict[key]

    def load_graph(self, graph_path):

        self.from_to_idxs = []
        self.data = []

        with open(graph_path, 'r') as graph_file:
            for line in graph_file:
                parsed_line = line.strip().split(' ')
                if len(parsed_line) in [2, 3]:
                    first = self._update_get_index(int(parsed_line[0]), self.from_dict)
                    second = self._update_get_index(int(parsed_line[1]), self.to_dict)
                    self.from_to_idxs.append([first, second])
                    if len(parsed_line) == 3:
                        self.data.append(int(parsed_line[2]))
                    else:
                        self.data.append(1)

        self.data = np.array(self.data).astype(np.float32)
        self.from_to_idxs = np.array(self.from_to_idxs).astype(np.int32)

    def save_mapping(self, output_dir):

        with open(os.path.join(output_dir, 'from_mapping.json')) as from_mapping_file:
            json.dump(self.from_dict, from_mapping_file)

        with open(os.path.join(output_dir, 'to_mapping.json')) as to_mapping_file:
            json.dump(self.to_dict, to_mapping_file)

    def fit(self, max_epochs=100, batch_size=1000, seed=1692, cost_tol=-np.inf):
        self.model = NodeVectorModel(
            n_from=len(self.from_dict),
            n_to=len(self.to_dict),
            de=self.dimensions
        )

        cost0 = np.inf
        for epoch_idx in xrange(max_epochs):

            random.seed(seed)
            random.shuffle(self.from_to_idxs)
            random.seed(seed)
            random.shuffle(self.data)

            seed = random.randint(0, 1e5)

            for obs_idx in xrange(0, len(self.data), batch_size):
                self.model.train(self.from_to_idxs[obs_idx:obs_idx + batch_size],
                                 self.data[obs_idx:obs_idx + batch_size])
                self.model.update_params(self.from_to_idxs[obs_idx:obs_idx + batch_size],
                                         self.data[obs_idx:obs_idx + batch_size])
                self.model.normalize(self.from_to_idxs[obs_idx:obs_idx + batch_size])

            cost1 = self.model.calculate_cost(self.from_to_idxs, self.data)
            if np.abs(cost0 - cost1) < cost_tol:
                return
            cost0 = cost1
            print cost1


if __name__ == "__main__":
    node2vec = Node2Vec(graph_path="data/edge.list", vector_dimensions=100)
    node2vec.save_mapping(output_dir="data")
    node2vec.fit(cost_tol=1e-10)
    node2vec.model.save_to_file("data/case_embeddings.pkl")
