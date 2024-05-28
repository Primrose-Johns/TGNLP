<h1>TGNLP:<br>A Three-Graph Natural Language Processing Library</h1>

<h2>Authors:<br>
Jesse Coulson:  <a href="https://www.linkedin.com/in/jessecoulson/">Linkedin</a>, <a href="https://github.com/jccoulson">Github</a><br>
Primrose Johns: <a href="https://www.linkedin.com/in/primrose-johns-4b957b15a/">Linkedin</a>, <a href="https://github.com/Primrose-Johns">Github</a></h2><br>

<!-- START OF INTRODUCTION -->
<h2>Introduction</h2>
This library allows users to create graphs that represent multiple kinds of word relationships in a provided corpus in Python3. Outputs are NetworkX graphs, allowing the user to use their powerful library to further investigate graph output. We also provide some functions of our own that produce subgraphs and metrics that tell the user more about a particular graph.<br>
Our work was originally inspired by <a href="https://github.com/THUMLP/TensorGCN">TensorGCN</a>.

<!-- START OF QUICK-START GUIDE -->
<h2>Quick Start Guide</h2>
<h3>Installation</h3>

You can install this library with Pypi, just run
```python
pip3 install TGNLP
```
to get the latest version!

<h3>Making a Corpus</h3>

To start, you'll need to build a `TGNLP.Corpus` object, which is what all graph generators take as input. The Corpus can be built from a Pandas Series of strings, a list of strings, or just one large string. A user can also provide a tokenized corpus as a list of lists of strings.

```python
import TGNLP as tgnlp

#data could be a pd.Series, a list, or one large string
data = "Some text."
corpus = tgnlp.Corpus(data)
```
`TGNLP.Corpus` objects are never modified by any functions that use them, so the same Corpus object can be reused to create as many graphs as are needed.

<h3>Generating a Graph</h3>

Once you have a `TGNLP.Corpus` object, you can use it to generate a graph. There are three types of graphs TGNLP can make, and you can generate them using `get_sequential_graph()`, `get_semantic_graph()`, and `get_syntactic_graph()`. The output is a weighted, undirected NetworkX graph.

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
This is done by trimming a percentage of the total edgeweight and <i>not</i> by trimming a certain percentage of all edges. This means that the trimming process may remove far more than 10% of the edges if a large portion of graph edges have very small weights. We recommend trimming at least 10% of edges by weight on all graphs, which is what `trim_norm_graph()` does by default. This function returns a trimmed copy of the provided graph be default, but you can use `inplace=True` to avoid the extra memory usage that entails.

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
The report also features visualizations of linear and logarithmic degree distributions, as well as overall graph metrics like average degree, and specific details on the highest-degree and lowest-degree nodes in the graph.<br><br><br><br>

<!-- START OF DOCUMENTATION -->

<h1>Documentation</h1>

<h2><i>Raw Text Parsing</i></h2>

We provide two methods of parsing raw text (or a raw text corpus) that a user would like to analyze. One is the `TGNLP.Corpus` class, which turns the user's raw text/text corpus into an object that our graph generators can use. The other is `dataframe_to_tokens_labels()`, which turns the raw text/text corpus into a tokenized format compatible with common feature extrapolation methods (bag or words, TF-IDF, word embeddings). The output of `dataframe_to_tokens_labels()` is a valid input to the `TGNLP.Corpus` class constructor.<br><br><br><br>

<!-- CORPUS CLASS -->
<h2>Corpus()</h2>

```python
Corpus(data)
```

<h4>Parameters</h4>

- **data**: *Pandas series, list, list of lists, or string*<br>
The data to be parsed. Lists, series, or the lists within lists passed in must contain only strings. Each string is expected to be at least as big as a single sentence, but can be as large as entire documents/collections of documents.

<h4>Returns</h4>

- **corpus**: *TGNLP Corpus object*<br>
A corpus representing the data parsed, which can be used for graph generation<br>

<h4>Attributes</h4>

- **sentence_corpus**: *List of strings*<br>
Every Sentence in the corpus in order of appearance. Each sentence is an item in the list. All punctuation and whitespace (except for one space between each word) has been removed.
- **word_corpus**: *List of strings*<br>
Every word in the corpus in order of appearance. Each word is an item in the list. All punctuation and whitespace has been removed.
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

<h2>dataframe_to_tokens_labels()</h2>

```python
dataframe_to_tokens_labels(df, text_column_name, label_column_name, lower_case=True, remove_stopwords=True, lemmatization = False)
```

<h4>Parameters</h4>

- **df**: *Pandas Dataframe*<br>
A DataFrame containing the text to be tokenized and their labels .
- **text_column_name**: *scalar object*<br>
The name of the column that contains the text to be tokenized.
- **label_column_name**: *scalar object*<br>
The Name of the column that contains the text labels.
- **lower_case**: *bool, default=True*<br>
Flag that determines whether or not to convert all words to lower case
- **remove_stopwords**: *bool, default=True*<br>
Flag that determines whether or not to remove stopwords.
- **lemmatization**: *bool, default=False*<br>
Flag that determines whether or not to remove lemmatize words.

<h4>Returns</h4>

- **document_list**: *List of list of strings*<br>
The text/corpus as lists of tokenized words.
- **label_list**: *List of \**<br>
The provided labels column, as a list.

<h4>Errors</h4>

- **TypeError** : *Inputted data is not of type Pandas DataFrame*<br>
Raised when the user-provided object is not a Pandas DataFrame.
- **ValueError** : *[text_column_name] is not a valid column in the DataFrame*<br>
Raised when the user-provided text column name is not found in the provided DataFrame.
- **ValueError** : *[label_column_name] is not a valid column in the DataFrame*<br>
Raised when the user-provided label column name is not found in the provided DataFrame.


This function turns two columns from a dataframe into preprocessed list of documents with each document containing a list of words. It optionally supports lemmatization, which will slow runtime significantly but provides more powerful results. The first output of this function, document_list, is a valid input type for the `TGNLP.Corpus` class constructor. <br><br><br><br>

<h2><i>The Three Graphs</i></h2>
TGNLP has three different graphs that it can generate, each of which represents different kinds of word relationships. All of the graphs are undirected, weighted graphs where nodes are represented by words and edges represent relationships between words. The weight of an edge represents how strong that relationship is, although what "strong" means depends on what graph is being worked with. There is the sequential graph, the semantic graph, and the syntactic graph.<br>These graphs were originally inspired by a methodology proposed by Xien Liu et al., and one implementation of their approach can be found in their <a href="https://github.com/THUMLP/TensorGCN">TensorGCN Github</a>.<br><br><br><br>

<!-- SEQUENTIAL GRAPH -->

<h2>get_sequential_graph()</h2>

```python
get_sequential_graph(corpus, window_size=5)
```
<h4>Parameters</h4>

- **corpus**: *TGNLP Corpus object*<br>
A corpus generated from the data to be analyzed
- **window_size**: *int, default=5*<br>
The size of the sliding window to be used for observing word co-occurrences
<h4>Returns</h4>

- **G**: *NetworkX Graph*<br>
A graph representing sequential relationships between words in the corpus
<h4>Errors</h4>

- **TypeError** : *Inputted data is not of type Corpus*<br>
Raised when the user provides a non-Corpus input

Nodes in the sequential graph represent individual words in the corpus. The weighted edges in the sequential graph represent how frequently two words appear near one another. This "nearness" is observed using a sliding window approach that the user can specify the size of. Every time two words appear in the same window, that counts as a co-occurrence of those two words. In an untrimmed graph every pair of words that appear together in a window will have an edge. The edge weight is calculated as $W_{i,j} = \frac{freq_{i,j}}{min\\{freq_{i}, freq_{j}\\}}$, where $freq_{i,j}$ is the number of co-occurences of two words and $min\\{freq_{i}, freq_{j}\\}$ is the overall frequency of the less frequent word.<br><br><br><br>


<!-- SEMANTIC GRAPH -->

<h2>get_semantic_graph()</h2>

```python
get_semantic_graph(corpus)
```
<h4>Parameters</h4>

- **corpus**: *TGNLP Corpus object*<br>
A corpus generated from the data to be analyzed
<h4>Returns</h4>

- **G**: *NetworkX Graph*<br>
A graph representing semantic relationships between words in the corpus
<h4>Errors</h4>

- **TypeError**: *Inputted data is not of type Corpus*<br>
Raised when the user provides a non-Corpus input 

This function uses word2vec to generate embeddings of each word. Once the embeddings are generated, we utilize the `most_similar(n)` function in Word2Vec in order to find the 20 most similar words to the provided word. These "most similar" word pairs each become an edge in the graph, with each node being a word. The cosine similarity between these words becomes the weight of the edge between them.<br><br><br><br>


<!-- SYNTACTIC GRAPH -->

<h2>get_syntactic_graph()</h2>

```python
get_syntactic_graph(corpus)
```
<h4>Parameters</h4>

- **corpus**: *TGNLP Corpus object*<br>
A corpus generated from the data to be analyzed
<h4>Returns</h4>

- **G**: *NetworkX Graph*<br>
A graph representing syntactic relationships between words in the corpus
<h4>Errors</h4>

- **TypeError**: *Inputted data is not of type Corpus*<br>
Raised when the user provides a non-Corpus input 

This function uses the spaCy library’s dependancy parser in order to identify all syntactic dependancies between wrods in each setences. While there are different types of syntactic dependancies in english, we treat all dependancy types as equal in terms of how much they contribute to edge weight. The edge weight is calculated as $W_{i,j} = \frac{freq_{i,j}}{min\\{freq_{i}, freq_{j}\\}}$, where $freq_{i,j}$ is the number of times two words share a dependancy in a sentence and $min\\{freq_{i}, freq_{j}\\}$ is the number of occurences of the less frequent word.<br><br><br><br>

<!-- GRAPH PROCESSING -->

<h2><i>Graph Processing</i></h2>
We offer a tool which normalizes edge weights and trims smaller edges.<br><br><br><br>

<h2>trim_norm_graph()</h2>

```python
trim_norm_graph(G_full, trim = 0.1, inplace = False)
```
<h4>Parameters</h4>

- **G_full**: *NetworkX Graph*<br>
A graph generated by TGNLP

- **trim**: *float, default=0.1*<br>
The amount of total edge weight to be trimmed. Must be between a positive value less than 1.

- **inplace**: *bool, default=False*<br>
Indicates whether or not to perform the trimming on the provided graph (False), or a copy of the graph (True).

<h4>Returns</h4>

- **G_proc**: *NetworkX Graph*<br>
A version of the provided graph with edges trimmed and normalized.
<h4>Errors</h4>

- **TypeError**: *Provided value for trim is too large. Trim value must be <= 1*<br>
Raised when the user provides trim value that is not a positive float less than 1

This function normalizes edge weight values in a provided graph to be between 0 and 1. It also trims a graph's edges to reduce the total weight of all edges. The latter process is done by iterating through each edge in ascending order of weight, and removing edges until the requested amount of the total edge weight has been removed. It will not remove an edge if doing so would remove *more* than the requested amount. We reccomend removing at least 10% (trim = 0.1) of edge weight in all of the graphs TGNLP generates.<br><br><br><br>


<h2><i>Graph Reporting</i></h2>
You can use our library to generate a report describing some important features of the graph you generated, and what it says about your text/corpus.<br><br><br><br>


<h2>generate_graph_report()</h2>

```python
generate_graph_report(G)
```
<h4>Parameters</h4>

- **G**: *NetworkX Graph*<br>
A graph generated by TGNLP

This function generates a pdf report (called tgnlp_report.pdf) that will appear in the directory from which the program is run. This report includes a visualization of a word subgraph (the word is chosen semi-randomly and must have a degree less than 15 and greater than 10), as well as some basic metrics about the graph in question. These metrics include the lowest-degree word, the highest-degree word, the number of nodes, the number of edges, the average degree, the average centrality, the assortativity coefficient, and both linear and logarithmic degree distributions.<br><br><br><br>

