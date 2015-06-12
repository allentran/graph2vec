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
                parsed_line = line.split(' ')
                if len(parsed_line) in [2, 3]:
                    max_index_in_line = np.max(parsed_line[:2])
                    if max_index_in_line > max_index:
                        max_index = max_index_in_line
                    self.from_to_idxes.append([parsed_line[0], parsed_line[1]])
                    if len(parsed_line) == 3:
                        self.data.append(parsed_line[2])
                    else:
                        self.data.append(1)

        self.n = max_index + 1

    def fit(self):

        self.model = NodeVectorModel(
            ne=self.n,
            de=self.dimensions
        )
