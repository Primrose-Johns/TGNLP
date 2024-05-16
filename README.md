<h1>TGNLP:<br>A Three-Graph Natural Language Processing Library</h1>

<h2>Authors:<br>
Jesse Coulson: <a href="https://www.linkedin.com/in/jessecoulson/">Linkedin</a>, <a href="https://github.com/jccoulson">Github</a><br>
Primrose Johns: <a href="https://www.linkedin.com/in/primrose-johns-4b957b15a/">Linkedin</a>, <a href="https://github.com/Primrose-Johns">Github</a></h2><br>
<h2>Introduction</h2>
This project allows users to create graphs that represent multiple kinds of word relationships in a provided corpus in Python3. Outputs are NetworkX graphs, allowing the user to use their powerful library to further investigate graph output. We also provide some functions of our own to provide subgraphs and metrics that tell the user more about a particular graph.<br>
Our work was originally inspired by <a href="https://github.com/THUMLP/TensorGCN">TensorGCN</a>.

<h2>Quick Start Guide</h2>

To start, you'll need to build a `tgnlp.Corpus()` object, which is what all graph generators take as input. The Corpus can be built from a Pandas Series of string, a list of strings, or just one large string.

```python
import TGNLP as tgnlp

#This could be a pd.Series, a list, or one large string
corpus = tgnlp.Corpus(data)
```

Once you have a Corpus object, you can use that to generate any of the graphs we have a generator for. The output is a weighted, undirected NetworkX graph.

```python
G = tgnlp.get_sequential_graph(corpus)
print(type(G))
```
Output:
```text
<class 'networkx.classes.graph.Graph'>
```

You can also trim the graph down, using `tgnlp.trim_norm_graph()`.<br>
```python
tgnlp.trim_norm_graph(G, inplace=True)
```
This is done by trimming a percentage of the total edgeweight and <i>not</i> by trimming a certain percentage of all edges. This means that the trimming process may remove far more than 10% of the edges. We reccomend trimming at least 10% of edges by weight, which is what `tgnlp.trim_norm_graph()` does by default.