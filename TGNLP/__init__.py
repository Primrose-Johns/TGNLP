import os
import pickle
import string
import pandas as pd
import re
import networkx as nx
from gensim.models import Word2Vec
import gensim
import gensim.downloader
import itertools
import timeit
import matplotlib.pyplot as plt
import re
from itertools import combinations
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import defaultdict
import spacy
from spacy.tokens import Doc
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from collections import Counter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
#python -m spacy download en_core_web_sm
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag




class Corpus:
  def __init__(self, data, lower_case = True, remove_stopwords = False, lemmatization = False):
    self.sentence_corpus = data_to_corpus(data, 'sentence', lower_case, remove_stopwords, lemmatization)
    self.word_corpus = data_to_corpus(data, 'word', lower_case, remove_stopwords, lemmatization)
    self.word_counts = get_word_counts(self.word_corpus)



###########################################
####Preprocessing Corpus Code##############
###########################################

def loadbbc():
    df = pd.read_csv("bbc_data.csv", header = 0)
    df["data"] = df["data"].str.replace(r"\S+xc2\S+", "", regex=True)
    return df["data"]


def check_list_of_lists_of_tokens(data):
    if isinstance(data, list):  
        for document in data:
            #check each item in big list is a list
            if isinstance(document, list):
                #check all words in inner list are strings  
                if all(isinstance(word, str) for word in document):
                    continue  
                return False 
            #not a list of lists
            else:
               return False
        #true if it is a list of list of strings
        return True  
    #false if outer structure is not a list
    return False 


#processes a string (or list/series of strings) for analysis
def data_to_corpus(data_raw, corpus_type='word', lower_case=True, remove_stopwords = False, lemmatization = False):
    #Load the data from the acceptable types
    if type(data_raw) == pd.core.series.Series:
        if data_raw.ndim != 1:
          raise ValueError("series must be 1 dimensional")
        temp = list(data_raw)
        if not check_type(temp):
          raise ValueError("series must only have elements of type: string")
        data = " ".join(temp)
    elif check_list_of_lists_of_tokens(data_raw) == True:
       if corpus_type == "sentence":
          #adding . to seperate documents
          temp_list = [" ".join(document) + "." for document in data_raw]
          data = " ".join(temp_list)
       else:
          #turn into string
          temp_list = [" ".join(document) for document in data_raw]
          data = " ".join(temp_list)
    elif type(data_raw) == list:
        if not check_type(data_raw):
            raise ValueError("list must only have elements of type: string")
        data = " ".join(data_raw)
    elif type(data_raw) == str:
        data = data_raw
    else:
        raise TypeError(f"Load_data requires one of type: Pandas Series, list of strings, list of list of tokens, string.\n  given data is of type{type(data_raw)}")

    
    #remove html tags
    data = re.sub(r"<.*?>", "", data)

    #remove numbers/anything with a number in it
    data = re.sub(r"\S+\d\S+", "", data)

 
    if corpus_type == "sentence":
        data = re.sub(r"[?!]", ".", data)
        data = re.sub(r"[^A-Za-z .]", "", data)
    elif corpus_type == "word":
        #remove special characters/punctuation
        data = re.sub(r"[^A-Za-z ]", "", data)

    #remove padding
    data = re.sub(r"\s+", " ", data).strip()

    #make all lowercase if specified
    if lower_case == True:
      data = data.lower()

    if lemmatization == True:
       #need to create dict to identify tags given by nltk pos_tag function
       pos_dict = {"J": nltk.corpus.wordnet.ADJ,
                  "N": nltk.corpus.wordnet.NOUN,
                  "V": nltk.corpus.wordnet.VERB,
                  "R": nltk.corpus.wordnet.ADV}
    if corpus_type == 'sentence':
        sentences = sent_tokenize(data)
        #check if need to remove stopwords or lemmatize
        if remove_stopwords == True or lemmatization == True:
            stop_words = set(stopwords.words('english'))
            cleaned_sentences = []
            #set lemmatizer here so it doesn't have to reinstatiate per sentence
            lemmatizer = None
            if lemmatization == True:
               lemmatizer = WordNetLemmatizer()
            #for each sentence remove stopwords/lemmatize if necissary
            for sentence in sentences:
                words = sentence.split()
                if remove_stopwords== True:
                    words = [word for word in words if word not in stop_words]
                if lemmatization == True:
                    tagged_words = pos_tag(words)
                    words = [lemmatizer.lemmatize(word,pos_dict.get(tag[0], 'n')) for word, tag in tagged_words]
                cleaned_sentence = ' '.join(words)
                cleaned_sentences.append(cleaned_sentence)
            return [re.sub("[^A-Za-z ]", "", sentence) for sentence in cleaned_sentences]
        else:
            return [re.sub("[^A-Za-z ]", "", sentence) for sentence in sentences]
    elif corpus_type == 'word':
        #split into list of words
        words = data.split()
        #remove stopwords
        if remove_stopwords == True:
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
        #lemmatize
        if lemmatization == True:
            lemmatizer = WordNetLemmatizer()
            tagged_words = pos_tag(words)
            #if word is unknown default to noun tag
            words = [lemmatizer.lemmatize(word,pos_dict.get(tag[0], 'n')) for word, tag in tagged_words]
        return words
    

#due to how this function is used, it assumes what is passed to it is a list
#helper function for load_data, used to make sure a list only has strings in it
def check_type(data):
  if not isinstance(data, list):
    raise TypeError("checktype must be given a list")
  return all(type(i)==str for i in data)



#create dictionary with words as key and occurences in corpus as value
#takes word corpus as input
def get_word_counts(corpus):
  words = {}
  for entry in corpus:
    if entry in words:
      words[entry] += 1
    else:
      words[entry] = 1
  return words


#helper function to process text from dataframe_to_tokens_labels
def process_text(text, lower_case=True, remove_stopwords=True, lemmatization = False):
  #remove html tags
  text = re.sub(r"<.*?>", "", text)
  
  #remove all special characters and punctuation
  text = re.sub(r"[^A-Za-z ]", "", text)

  if lower_case == True:
      text = text.lower()

  #turn into list of words
  text = text.split()

  if remove_stopwords == True:
     stop_words = set(stopwords.words('english'))
     text = [word for word in text if word not in stop_words] 

  if lemmatization == True:
    pos_dict = {"J": nltk.corpus.wordnet.ADJ,
          "N": nltk.corpus.wordnet.NOUN,
          "V": nltk.corpus.wordnet.VERB,
          "R": nltk.corpus.wordnet.ADV}
    lemmatizer = WordNetLemmatizer()
    tagged_words = pos_tag(text)
    text= [lemmatizer.lemmatize(word,pos_dict.get(tag[0], 'n')) for word, tag in tagged_words]
  
  return text
     
#turn a dataframe into preprocessed list of documents with each document containing list of words
def dataframe_to_tokens_labels(df, text_column_name, label_column_name, lower_case=True, remove_stopwords=True, lemmatization = False):
  #checking if valid dataframe
  if not isinstance(df, pd.DataFrame):
      raise TypeError("Inputted data is not of type Pandas DataFrame")
  
  #checking that columns exist in the dataframe
  if text_column_name not in df.columns:
      raise ValueError(f"{text_column_name} is not a valid column in the DataFrame")
  if label_column_name not in df.columns:
      raise ValueError(f"{label_column_name} is not a valid column in the DataFrame")
  
  #drop missing values in either columns
  df.dropna(subset=[text_column_name, label_column_name], inplace=True)

  #list to hold all documents
  document_list = []
  for document in df[text_column_name]:
    #process each document
    document_list.append(process_text(document,lower_case, remove_stopwords, lemmatization))

  #turn column of labels into list
  label_list = df[label_column_name].to_list()

  #return the two lists, will be in matching order
  return document_list, label_list





###########################################
####Syntactic Graph Code###################
###########################################

#creates the graph from preprocessed data
def get_syntactic_graph(tgnlp_corpus):
    if not isinstance(tgnlp_corpus, Corpus):
      raise TypeError("Inputted data is not of type Corpus") 

    sentence_corpus = tgnlp_corpus.sentence_corpus
    word_counts = tgnlp_corpus.word_counts

    #returns edge list as dict in form (word1, word2) : weight
    edge_dict = syntactic_pair_generation(sentence_corpus)

    #returns networkx graph
    return syn_graph_generation(edge_dict, word_counts)




def syntactic_pair_generation(sentence_corpus):
    #default dict so that checking if key exists is not necissary
    edge_dict = defaultdict(int)

    #needs download python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    #creating custom tokenizer because we already preprocessed corpus
    def custom_tokenizer(text):
      tokens = text.split()
      return Doc(nlp.vocab, words=tokens)

    nlp.tokenizer = custom_tokenizer

    for sentence in sentence_corpus:
       for word in nlp(sentence):
          #takes care of root
          if word.text == word.head.text:
             continue
          
          #adding one weight for each syntactic dependcy, could further addd different weights depending on type of dependency
          edge_dict[(word.text, word.head.text)] +=1

    return edge_dict

def syn_graph_generation(edge_dict, word_counts):
    #create graph
    G = nx.Graph()
    for (node1, node2), weight in edge_dict.items():
      normalized_weight = weight/min(word_counts[node1], word_counts[node2])

      G.add_edge(node1, node2, weight=normalized_weight)
    return G

            


###########################################
####Semantic Graph Code####################
###########################################

def get_semantic_graph(tgnlp_corpus):
    if not isinstance(tgnlp_corpus, Corpus):
      raise TypeError("Inputted data is not of type Corpus") 

    word_corpus = tgnlp_corpus.word_corpus
    sentence_corpus = tgnlp_corpus.sentence_corpus
    word_counts = tgnlp_corpus.word_counts

    #format for word2vec input, takes list of list of words, outer list is sentences inner list is words in setnences
    word2vec_input = []
    for sentence in sentence_corpus:
        temp_word_list = []
        sentence_list = sentence.split(' ')
        for word in sentence_list:
            temp_word_list.append(word)
        word2vec_input.append(temp_word_list)

    #using skipgram, better for small corpus
    model = gensim.models.Word2Vec(word2vec_input, min_count=1, vector_size=50, window=5, sg=1)

    #returns edge list in form [(word1, word2, weight)...]
    edge_list = generate_sem_edgelist(model, word_corpus, sentence_corpus, word_counts)

    #return semantic graph
    return sem_graph_generation(edge_list)


def sem_graph_generation(edge_list):
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)
    return G

def generate_sem_edgelist(model, word_corpus, sentence_corpus, word_counts):
    sem_edgelist = []
    
    #unique list of words
    unique_word_list = list(set(word_corpus))
    for word1 in unique_word_list:
        similar = model.wv.most_similar(word1, topn = 20)

        #tuple of word and cosine similarity
        for word_similarity in similar:
            word2 = word_similarity[0]

            #word2vec can return empty string, need to make sure not to use
            if len(word2) == 0:
               continue
              
            #taking out normalization for semntic
            #norm_edgeweight = word_similarity[1]/(min(word_counts[word1], word_counts[word2]))
            sem_edgelist.append((word1, word2, word_similarity[1]))
  
    return sem_edgelist


###########################################
####Sequential Graph Code##################
###########################################

#returns a networkx graph representing sequential relationships between words
#in the provided corpus
def get_sequential_graph(tgnlp_corpus, window_size=5):
  
  if not isinstance(tgnlp_corpus, Corpus):
    raise TypeError("Inputted data is not of type Corpus") 

  corpus = tgnlp_corpus.word_corpus
  word_counts = tgnlp_corpus.word_counts
  
  G = nx.Graph()
  #get the dict of word pair occurences
  pairs = sliding_window(corpus, window_size)
  #for each entry
  for key, val in pairs.items():
    #deconstruct dict keys (get both words)
    first, second = key.split(",")
    denom = min([word_counts[first], word_counts[second]])
    weight = val/denom
    G.add_edge(first, second, weight = weight)
    #print("Edge added")
    #create a weighted edge from word A to word B
  #print(f"In seq function: Graph nodes = {G.number_of_nodes()}")
  return G

def sliding_window(corpus, window_size):
  #word pair co-occurences
  occ = {}
  num_windows = len(corpus)-(window_size-1)
  for i in range(num_windows):
    j = i+window_size
    sub_range = corpus[i:j]
    pairs = combinations(sub_range, 2)
    #update word pair co-occurences
    for pair in pairs:
      #create the word-pair key
      first, second = pair
      if first == second:
        continue
      if first < second:
        key = first+","+second
      else:
        key = second+","+first
      #update the dict
      if key in occ:
        occ[key] += 1
      else:
        occ[key] = 1
  return occ


###########################################
####Normalization Code######################
###########################################

def trim_norm_graph(G_full, trim = 0.1, inplace = False):
    if trim > 1:
      raise ValueError("Provided value for trim is too large. Trim value must be <= 1")
    if inplace:
      G = G_full
    else:
      G = deepcopy(G_full) 
    #print("deepcopy done")
    #scaling portion, this will normalize the values between 0 and 1
    #Grabs the edges as a big list with format [node1, node2, dict of attributes]
    edges = list(G.edges(data=True))
    weights = [e[2]["weight"] for e in edges]
    weights = np.array(weights).reshape(-1, 1)
    scaler = MinMaxScaler()
    #here's where we get our new weights, between 0 and 1
    weights = scaler.fit_transform(weights)
    #reshaping this again to make the values easier to access
    weights = np.array(weights).reshape(-1)
    i = 0
    #print("entering main scaling loop")
    for a, b, d in edges:
        #the edges and weights objects represent the same edges in the same order
        new_weight = weights[i]
        #print("weight tranformed")
        #assign the new weight within the graph
        G[a][b]["weight"] = new_weight
        #print("graph weight assigned")
        #assign the new weight within the "edges" and "weights" lists
        edges[i][2]["weight"] = new_weight
        #print("edges list updated")
        i+=1
    #print("Scaling done")
    #Triming portion
    #simple integral (no width of rectangle needs to be considered)
    auc = sum(weights)
    trim_amount = auc*trim
    trimmed = 0
    #sort out list of scaled edges in ascending order
    edges = sorted(edges, key=lambda val : val[2]["weight"])
    #remove nodes until trim_amount of weights have been removed
    for a, b, d in edges:
        weight = d["weight"]
        if trimmed+weight < trim_amount:
            G.remove_edge(a, b)
            trimmed += weight
        else:
            break
    #print("trimming done")
    return G


###########################################
####Graph Analysis Functions###############
###########################################

def word_subgraph(G, word, depth=1):
  #initialize set with original word
  node_set = set([word])

  #add neighbors to set, set will be used to track all nodes for nbunch in subgraph
  current_neighbors = G.neighbors(word)
  node_set.update(list(current_neighbors))

  for i in range(depth-1):
     build_neighbors(G, node_set)

  #return subgraph with all nodes found
  return G.subgraph(list(node_set))

def build_neighbors(G, node_set):
  found_nodes = []
  for node in node_set:
     found_nodes.extend(list(G.neighbors(node)))
  
  #return set with all new neighbors found
  node_set.update(found_nodes)
  return node_set

###########################################
####Graph Report Functions#################
###########################################

def generate_graph_report(G):
  #get all metrics to be dispalyed in report
  metrics_dict = generate_metrics(G)

  #create document with USA standard of letter
  doc = SimpleDocTemplate("tgnlp_report.pdf", pagesize=letter)

  #holding all elements to add to pdf
  elements = []

  #create title
  styles = getSampleStyleSheet()
  style = styles['Title']
  title = Paragraph("TGNLP Network Report", style)
  
  #appending elements to list in order of displaying
  elements.append(title)
  elements.append(Spacer(1, 20))

  #graph metrics for first table
  graph_data = [
     ["Graph Metrics"],
     ["number of nodes", metrics_dict["number_of_nodes"]],
     ["number of edges", metrics_dict["number_of_edges"]],
     ["average degree", format(metrics_dict["average_degree"], ".4f")], #displaying up to four decimal places
     ["average degree centrality", format(metrics_dict["average_centrality"], ".4f")],
     ["assortativity coefficient", format(metrics_dict["assortativity_coefficient"], ".4f")],
     ["highest degree word", metrics_dict["highest_degree_word"][0]],
     ["lowest degree word", metrics_dict["lowest_degree_word"][0]],
  ]

  #word metrics for second table
  word_data = [
     ["Word", "Degree", "Degree Centrality"],
     [metrics_dict["highest_degree_word"][0], metrics_dict["highest_degree_word"][1], format(metrics_dict['highest_degree_word_centrality'], ".4f")], 
     [metrics_dict["lowest_degree_word"][0], metrics_dict["lowest_degree_word"][1],format(metrics_dict['lowest_degree_word_centrality'], ".4f")],
  ]

  #create first table
  table1 = Table(graph_data, colWidths = [200,200])
  table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 13),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        #for header row make it centered
        ('SPAN', (0, 0), (-1, 0)),
  ]))

  #create second table
  table2 = Table(word_data, colWidths = [133,133,133])
  table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 13),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
  ]))

  elements.append(table1)
  elements.append(Spacer(1, 13))
  elements.append(table2)

  #making figures of degree distributions
  fig_buffer1, fig_buffer2 = generate_degree_distribution(metrics_dict["degree_count"])
  fig1 = Image(fig_buffer1, width=500, height=300)
  fig2 = Image(fig_buffer2, width=500, height=300)
  elements.append(fig1)
  elements.append(fig2)

  #displaying a subgraph of a word in the graph
  subword = ''

  #finding a word with 10-15 degrees to display, good number for clear visualization
  for node, degree in G.degree():
     if degree >= 10 and degree <= 15:
        subword = node
        break

  #catching edge cases where network has no words in degree range  
  if subword == '':
    word, degree = metrics_dict["highest_degree_word"]

    #if the highest degree is less than 10 just visualize the highest degree word
    if degree < 10:
       subword = word
    else:
    #if greater than 15 just visualize the lowest degree word
       subword = metrics_dict["lowest_degree_word"][0]
  
  #get word subgraph
  word_subg = word_subgraph(G, subword)

  #plot figure
  plt.figure(figsize=(12, 8))
  plt.title(f"Example Word \"{subword}\" Subgraph")
  pos = nx.spring_layout(word_subg)
  nx.draw_networkx(word_subg, pos,with_labels=True, node_color = 'lightblue', edge_color = 'lightgray', node_size = [100 * word_subg.degree(node) for node in word_subg.nodes()])

  #save to buffer
  wordsub_buffer = BytesIO()
  plt.savefig(wordsub_buffer, format='PNG')
  plt.clf()

  #add to pdf
  fig3 = Image(wordsub_buffer, width=500, height=300)
  elements.append(fig3)

  #build the final pdf
  doc.build(elements)

def generate_degree_distribution(deg_counts):
  #plot degree distribution
  plt.figure(figsize=(12, 8))
  plt.scatter([key for key in deg_counts.keys()], [val for val in deg_counts.values()], color = 'b')
  plt.title("Degree Distribution of Network")
  plt.ylabel('Number of Nodes')
  plt.xlabel('Degree')
  #save fig to buffer
  fig_buffer1 = BytesIO()
  plt.savefig(fig_buffer1, format='PNG')

  #plot degree distribution in log log space
  plt.xscale("log")
  plt.yscale("log")
  plt.title("Degree Distribution of Network(log)")
  plt.ylabel('Number of Nodes(log)')
  plt.xlabel('Degree(log)')
  fig_buffer2 = BytesIO()
  plt.savefig(fig_buffer2, format='PNG')

  plt.clf()
  return fig_buffer1, fig_buffer2
  

def generate_metrics(G):
  #dict to hold all metrics
  metrics_dict = {}

  #creating list of sorted tuples of node,degree  by degree
  sorted_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)

  #first and last words in sorted list
  highest_degree_word = sorted_degree[0][0]
  lowest_degree_word = sorted_degree[-1][0]

  #getting general metrics
  metrics_dict['number_of_nodes'] = G.number_of_nodes()
  metrics_dict['number_of_edges'] = G.number_of_edges()

  metrics_dict['highest_degree_word'] = (highest_degree_word,sorted_degree[0][1])
  metrics_dict['lowest_degree_word'] = (lowest_degree_word,sorted_degree[-1][1])
  metrics_dict['average_degree'] = sum([degree[1] for degree in G.degree]) / G.number_of_nodes()
  
  centrality_dict =  nx.degree_centrality(G)
  metrics_dict['highest_degree_word_centrality'] = centrality_dict[highest_degree_word]
  metrics_dict['lowest_degree_word_centrality'] = centrality_dict[lowest_degree_word]
  metrics_dict['average_centrality'] = sum(centrality_dict.values()) / G.number_of_nodes()
  
  metrics_dict["assortativity_coefficient"] = nx.degree_assortativity_coefficient(G)
 
  #creating degree sequence
  degree_list = [degree for _, degree in sorted_degree]

  #using collections to get count of each degree
  degree_count = Counter(degree_list)
  metrics_dict["degree_count"] = degree_count

  return metrics_dict
