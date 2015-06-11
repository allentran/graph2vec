__author__ = 'porky-chu'

import numpy as np
import scipy.sparse

class NodeVectors(object):

    def __init__(self, alpha, d_vectors, learning_rate):

        self.alpha = alpha
        self.d_vectors = d_vectors
        self.learning_rate = learning_rate

        self.n
        self.X = None
        self.Win = None
        self.Wout = None

    def load_graph(self, graph_path):

        columns = []
        rows = []
        data = []
        max_index = 0

        with open(graph_path, 'r') as graph_file:
            for line in graph_file:
                parsed_line = line.split(' ')
                if len(parsed_line) in [2, 3]:
                    max_index_in_line = np.max(parsed_line[:2])
                    if max_index_in_line > max_index:
                        max_index = max_index_in_line
                    rows.append(parsed_line[0])
                    columns.append(parsed_line[1])
                    if len(parsed_line) == 3:
                        data.append(parsed_line[2])
                    else:
                        data.append(1)

        self.n = max_index + 1
        self.X = scipy.sparse.csr_matrix((data, (rows, columns)), shape=(max_index + 1, max_index + 1))

    def _init_vectors(self):

        self.Win = np.random.uniform(low=-1, high=1, size=(self.n, self.d_vectors))
        self.Wout = np.random.uniform(low=-1, high=1, size=(self.n, self.d_vectors))

    def train(self):

        self._init_vectors()
