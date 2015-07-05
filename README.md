# graph2vec
Learning node embeddings via Theano (the GPU if you have a good one), with minibatch and AdaGrad.

Install via running
```
python setup.py install
```
or
```
pip install graph2vec
```
Data should be in space delimited files describing edges, either as `from_node to_node distance` or without the path distance.  For example, in a text file like
```
0 1
1 3
14 21
21 1
```
Use the helper `trainer` module to build/load the graph.
```
import graph2vec.trainer

graph2vec = Graph2Vec(vector_dimensions=128)
graph2vec.parse_graph('edge.data', extend_paths=2)
```
The `extend_paths` argument dictates the length of paths included in the cost function.  Note that only the shortest path between nodes are included.  Fit the vectors with
```
graph2vec.fit(batch_size=1000, max_epochs=1000)
```
Don't go too crazy with the batch size, you'll get a speed up but convergence will be erratic. The vectors, both as origin and destination nodes are stored in the model object.
```
graph2vec.model.Win.get_value()
graph2vec.model.Wout.get_value()
```

