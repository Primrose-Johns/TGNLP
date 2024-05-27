<h1>TGNLP:<br>A Three-Graph Natural Language Processing Library</h1>

<h2>Authors:<br>
Jesse Coulson: <a href="https://www.linkedin.com/in/jessecoulson/">Linkedin</a>, <a href="https://github.com/jccoulson">Github</a><br>
Primrose Johns: <a href="https://www.linkedin.com/in/primrose-johns-4b957b15a/">Linkedin</a>, <a href="https://github.com/Primrose-Johns">Github</a></h2><br>

<!-- START OF INTRODUCTION -->
<h2>Introduction</h2>
This project allows users to create graphs that represent multiple kinds of word relationships in a provided corpus in Python3. Outputs are NetworkX graphs, allowing the user to use their powerful library to further investigate graph output. We also provide some functions of our own that produce subgraphs and metrics that tell the user more about a particular graph.<br>
Our work was originally inspired by <a href="https://github.com/THUMLP/TensorGCN">TensorGCN</a>.

<!-- START OF QUICK-START GUIDE -->
<h2>Quick Start Guide</h2>
<h3>Making a Corpus</h3>

To start, you'll need to build a `Corpus()` object, which is what all graph generators take as input. The Corpus can be built from a Pandas Series of strings, a list of strings, or just one large string.

```python
import TGNLP as tgnlp

#data could be a pd.Series, a list, or one large string
corpus = tgnlp.Corpus(data)
```
Corpuses are never modified by any functions that use them, so the same Corpus object can be reused to create as many graphs as are needed.

<h3>Generating a Graph</h3>

Once you have a Corpus object, you can use it to generate the graphs we have implemented using `get_sequential_graph()`, `get_semantic_graph()`, and `get_syntactic_graph()`. The output is a weighted, undirected NetworkX graph.

```python
G = tgnlp.get_sequential_graph(corpus)
print(type(G))
```
Output:
```text
<class 'networkx.classes.graph.Graph'>
```
<h3>Graph Processing</h3>

You can also trim the graph down, using `trim_norm_graph()`.<br>
```python
tgnlp.trim_norm_graph(G, inplace=True)
```
This is done by trimming a percentage of the total edgeweight and <i>not</i> by trimming a certain percentage of all edges. This means that the trimming process may remove far more than 10% of the edges if a large portion of graph edges have very small weights. We reccomend trimming at least 10% of edges by weight on all graphs, which is what `trim_norm_graph()` does by default. By default the trimming is done on a copy of the provided graph, but you can use `inplace=True` to avoid the extra memory usage that entails.

<h3>Graph Analysis</h3>

You can get a PDF report summarizing some of the important metrics about your corpus using `generate_graph_report()`, with your graph as input. It will appear in the directory the script is called from with the name `tgnlp_report.pdf`.
```python
#This will show up in the directory the python script is called from 
tgnlp.generate_graph_report(G)
```
The report is a two pages long, here's an example of one of our word subgraphs<br>
<p align="center">
<img src="Example_Report.png" style="width: 55vw">
</p>
The report also features visualizations of linear and logarithmic degree distributions, as well as overall graph metrics like average degree, and specific details on the highest- and lowest-degree nodes in the graph.<br><br><br><br>

<!-- START OF DOCUMENTATION -->

<h1>Documentation</h1>

<h2><i>Raw Text Parsing</i></h2>

We provide two methods of parsing raw text (or a raw text corpus) that a user would like to analyze. One is the Corpus class, which turns the users raw text/text corpus into an object that our graph generators can use. The other is `dataframe_to_tokens_labels()`, which turns raw text or a raw text corpus into a tokenized format compatible with common feature extrapolation methods (bag or words, TF-IDF, word embeddings).<br><br><br><br>

<!-- CORPUS CLASS -->
<h2>Corpus()</h2>

```python
Corpus(data)
```

<h4>Parameters</h4>

- **data**: *Pandas series, list, or string*<br>
The data to be parsed. Lists or series passed in must contain only strings

<h4>Returns</h4>

- **corpus**: *TGNLP Corpus object*<br>
A corpus representing the data parsed, which can be used for graph generation<br>

<h4>Attributes</h4>

- **sentence_corpus**: *List of strings*<br>
Every Sentence in the corpus in order of appearance. Each sentence is an item in the list. All punctuation and whitespace has been removed.
- **word_corpus**: *List of strings*<br>
Every word in the collective corpus in order of appearance. Each word is an item in the list. All punctuation and whitespace has been removed.
- **word_counts**: *Dict of string:int*<br>
Every word in the corpus is a key in the dict. Values are the number of appearances that word makes over the whole corpus.<br>

<h4>Errors</h4>

- **TypeError**: *Load_data requires one of type: Pandas Series, list of strings, list of list of tokens, string.*<br>
Raised when the user provides a data input of incorrect type.
- **ValueError**: *series must only have elements of type: string*<br>
Raised when the user provides a Pandas Series that is not populated with exclusively strings.
- **ValueError**: *list must only have elements of type: string*<br>
Raised when the user provides a list that is not populated with exclusively strings.


This is the object that all graph generation functions take as input. It stores a word corpus, a sentence corpus, and a word frequency dictionary. Corpus objects are never modified or altered by any TGNLP functions after they are created, so they can be reused to create multiple graphs as needed.<br><br><br><br>

\<dataframe_to_tokens_labels() goes here!\>

<h2><i>The Three Graphs</i></h2>
TGNLP has three different graphs that it can generate, each of which represents different kinds of word relationships. All of the graphs are undirected, weighted graphs where nodes are represented by words and edges represent relationships between words. The weight of an edge represents how strong that relationship is, although what "strong" means depends on what graph is being worked with. There is the sequential graph, the semantic graph, and the syntactic graph.<br>These graphs were originally inspired by a methodology proposed by Xien Liu et al., and one implementation of their approach can be found in their <a href="https://github.com/THUMLP/TensorGCN">TensorGCN Github</a>.<br><br><br><br>

<!-- SEQUENTIAL GRAPH -->

<h2>get_sequential_graph()</h2>

```python
get_sequential_graph(corpus, window_size=5)
```
<h4>Parameters</h4>

- **corpus**: *TGNLP Corpus*<br>
A corpus generated from the data to be analyzed
- **window_size**: *int, default=5*<br>
The size of the sliding window to be used
<h4>Returns</h4>

- **G**: *NetworkX Graph*<br>
A graph representing sequential relationships between words in the corpus
<h4>Errors</h4>

- **TypeError** : *Inputted data is not of type Corpus*<br>
Raised when the user provides a non-Corpus input

Nodes in the sequential graph represent individual words in the corpus. The weighted edges in the sequential graph represent how frequently two words appear near one another. This "nearness" is observed using a sliding window approach that the user can specify the size of. Every time two words appear in the same window, that counts towards the weight of their edge. In an untrimmed graph every pair of words that appear together in a window will have an edge. The edge weight is calculated as $W_{i,j} = \frac{freq_{i,j}}{min\\{freq_{i}, freq_{j}\\}}$, where $freq_{i,j}$ is the number of co-occurences of two words in the same window and $min\\{freq_{i}, freq_{j}\\}$ is the frequency of the less frequent word.<br><br><br><br>


<!-- SEMANTIC GRAPH -->

<h2>get_semantic_graph()</h2>

```python
get_semantic_graph(corpus)
```
<h4>Parameters</h4>

- **corpus**: *TGNLP Corpus*<br>
A corpus generated from the data to be analyzed
<h4>Returns</h4>

- **G**: *NetworkX Graph*<br>
A graph representing semantic relationships between words in the corpus
<h4>Errors</h4>

- **TypeError**: *Inputted data is not of type Corpus*<br>
Raised when the user provides a non-Corpus input 

This function uses word2vec to generate embeddings of each word. Nodes are words, and edges indicate a semantic relationship between words. We use Word2Vec to generate embeddings of words in the provided corpus. Once the embeddings are generated, we utilize the `most_similar(n)` function in Word2Vec
in order to find the 20 most similar words to the provided word. The cosine similarity between these words becomes the weight of the edge between them, which can be normalized if necessary.<br><br><br><br>


<!-- SYNTACTIC GRAPH -->

<h2>get_syntactic_graph()</h2>

```python
get_syntactic_graph(corpus)
```
<h4>Parameters</h4>

- **corpus**: *TGNLP Corpus*<br>
A corpus generated from the data to be analyzed
<h4>Returns</h4>

- **G**: *NetworkX Graph*<br>
A graph representing syntactic relationships between words in the corpus
<h4>Errors</h4>

- **TypeError**: *Inputted data is not of type Corpus*<br>
Raised when the user provides a non-Corpus input 

This function uses the spaCy library’s dependancy parser in order to identify syntactic dependancies in setences. While there are different types of syntactic dependancies in english, we treat all dependancy types as equal in terms of how much they contribute edge weight. The edge weight is calculated as $W_{i,j} = \frac{freq_{i,j}}{min\\{freq_{i}, freq_{j}\\}}$, where $freq_{i,j}$ is the number of times two words share a dependancy in a sentence and $min\\{freq_{i}, freq_{j}\\}$ is the number of occurences of the less frequent word.<br><br><br><br>

<!-- GRAPH PROCESSING -->

<h2><i>Graph Processing</i></h2>
We offer a tool which normalizes edge weights and trims smaller edges.<br><br><br><br>

<h2>trim_norm_graph()</h2>

```python
trim_norm_graph(G_full, trim = 0.1, inplace = False)
```
<h4>Parameters</h4>

- **G**: *NetworkX Graph*<br>
A graph generated by TGNLP

- **trim**: *float, default=0.1*<br>
The amount of edges to be trimmed, by edge weight. Must be between a positive value less than 1.

- **inplace**: *bool, default=False*<br>
Indicates whether or not to perform the trimming on the provided graph, or a copy of the graph. Python's `deepcopy()` is used to create the copy.

<h4>Returns</h4>

- **G_proc**: *NetworkX Graph*<br>
A trimmed, normalized version of the provided graph.
<h4>Errors</h4>

- **TypeError**: *Provided value for trim is too large. Trim value must be <= 1*<br>
Raised when the user provides trim value that is not a positive float less than 1

This function normalizes edge weight values in a provided graph to be between 0 and 1. It also trims a graph's edges to reduce the total weight of all edges. The latter process is done by iterating through each graph edge in ascending order of weight, and removing edges until the requested amount of the total edge weight has been removed. It will not remove an edge if doing so would remove *more* than the requested amount. We reccomend removing at least 10% (trim = 0.1) of edges in all of the graphs TGNLP generates.


<h2><i>Graph Reporting</i></h2>
You can use our library to generate a report describing some important features of the graph you generated, and what it says about your text/corpus.<br><br><br><br>
