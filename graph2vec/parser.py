__author__ = 'allentran'

import json
import os
import multiprocessing

import numpy as np

def _update_min_dict(candidate_node, depth, min_set):
    if candidate_node in min_set:
        if min_set[candidate_node] <= depth:
            return
        else:
            min_set[candidate_node] = depth
    else:
        min_set[candidate_node] = depth

def _get_connected_nodes((node_idx, adjancency_list, max_degree), current_depth=1):
    connected_dict = {}
    single_degree_nodes = [other_idx for other_idx in adjancency_list[node_idx] if adjancency_list[node_idx][other_idx] == 1]
    for other_idx in single_degree_nodes:
        _update_min_dict(other_idx, current_depth, connected_dict)

    if current_depth <= max_degree:
        for other_node_idx in single_degree_nodes:
            if other_node_idx in adjancency_list:
                new_connected_nodes = _get_connected_nodes((other_node_idx, adjancency_list, max_degree), current_depth + 1)
                if new_connected_nodes is not None:
                    for other_idx, depth in new_connected_nodes.iteritems():
                        _update_min_dict(other_idx, depth, connected_dict)
        return connected_dict

class Graph(object):

    def __init__(self, graph_path):

        self.from_nodes_mapping = {}
        self.to_nodes_mapping = {}

        self.edge_dict = {}

        self._load_graph(graph_path=graph_path)
        self._create_mappings()

    def save_mappings(self, output_dir):

        with open(os.path.join(output_dir, 'from.map'), 'w') as from_map_file:
            json.dump(self.from_nodes_mapping, from_map_file)
        with open(os.path.join(output_dir, 'to.map'), 'w') as to_map_file:
            json.dump(self.to_nodes_mapping, to_map_file)

    def get_mappings(self):
        return self.from_nodes_mapping, self.to_nodes_mapping

    def _create_mappings(self):
        for key in self.edge_dict:
            self.from_nodes_mapping[key] = len(self.from_nodes_mapping)
        for to_nodes in self.edge_dict.values():
            for to_node in to_nodes:
                if to_node not in self.to_nodes_mapping:
                    self.to_nodes_mapping[to_node] = len(self.to_nodes_mapping)

    def _add_edge(self, from_idx, to_idx, degree=1):
        if from_idx not in self.edge_dict:
            self.edge_dict[from_idx] = dict()
        if to_idx in self.edge_dict[from_idx]:
            if degree >= self.edge_dict[from_idx][to_idx]:
                return
        self.edge_dict[from_idx][to_idx] = degree

    def _load_graph(self, graph_path):

        with open(graph_path, 'r') as graph_file:
            for line in graph_file:
                parsed_line = line.strip().split(' ')
                if len(parsed_line) in [2, 3]:
                    from_idx = int(parsed_line[0])
                    to_idx = int(parsed_line[1])
                    if len(parsed_line) == 3:
                        degree = int(parsed_line[2])
                        self._add_edge(from_idx, to_idx, degree)
                    else:
                        self._add_edge(from_idx, to_idx)

    def extend_graph(self, max_degree, penalty=2):

        def _zip_args_for_parallel_fn():
            for key in self.from_nodes_mapping.keys():
                yield (key, self.edge_dict, max_degree)

        from_to_idxs = []
        degrees = []

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        connected_nodes_list = pool.map(_get_connected_nodes, _zip_args_for_parallel_fn())
        pool.close()
        pool.join()

        for node_idx, connected_nodes in zip(self.from_nodes_mapping.keys(), connected_nodes_list):
            for other_node, degree in connected_nodes.iteritems():
                from_to_idxs.append([self.from_nodes_mapping[node_idx], self.to_nodes_mapping[other_node]])
                degrees.append(float(1)/(degree ** penalty))

        return np.array(from_to_idxs).astype(np.int32), np.array(degrees).astype(np.float32)
