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

import nltk
#nltk.download('punkt')
#python -m spacy download en_core_web_sm
from nltk.tokenize import sent_tokenize, word_tokenize




class Corpus:
  def __init__(self, data):
    self.sentence_corpus = data_to_corpus(data, 'sentence')
    self.word_corpus = data_to_corpus(data, 'word')
    self.word_counts = get_word_counts(self.word_corpus)



###########################################
####Preprocessing Corpus Code##############
###########################################
def loadbbc():
    df = pd.read_csv("bbc_data.csv", header = 0)
    df["data"] = df["data"].str.replace(r"\S+xc2\S+", "", regex=True)
    return df["data"]


#processes a string (or list/series of strings) for analysis
def data_to_corpus(data_raw, corpus_type='word'):
    #Load the data from the acceptable types
    if type(data_raw) == pd.core.series.Series:
        print("Series")
        assert data_raw.ndim == 1, "series must be 1 dimensional"
        temp = list(data_raw)
        print(len(temp[0]))
        assert check_type(temp), "series must only have elements of type: string"
        data = " ".join(temp)
        print(len(data))
    elif type(data_raw) == list:
        print("List")
        assert check_type(data_raw), "list must only have elements of type: string."
        data = " ".join(data_raw)
    elif type(data_raw) == str:
        print("string")
        data = data_raw
    else:
        assert False, f"Load_data requires one of type: Pandas Series, list of strings, string.\n  given data is of type{type(data_raw)}"
    #clean the data
    #remove numbers/anything with a number in it
    data = re.sub("\S+\d\S+", "", data)

 
    if corpus_type == 'sentence':
        data = re.sub("[^A-Za-z .]", "", data)
    elif corpus_type == 'word':
        #remove special characters/punctuation
        data = re.sub("[^A-Za-z ]", "", data)

    #remove padding
    data = re.sub("\s+", " ", data).strip()
    #make all lowercase
    data = data.lower()


    if corpus_type == 'sentence':
        #split into sentences with nltk
        sentences = sent_tokenize(data)

        #remove "." from all sentences
        sentences = [re.sub("[^A-Za-z ]", "",sentence) for sentence in sentences]   
        return sentences
    elif corpus_type == 'word':    
        return data.split(' ')
    
    

#due to how this function is used, it assumes what is passed to it is a list
#helper function for load_data, used to make sure a list only has strings in it
def check_type(data):
  assert (type(data) == list), "checktype must be given a list"
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



###########################################
####Syntactic Graph Code###################
###########################################



#creates the graph from preprocessed data
def get_syntactic_graph(tgnlp_corpus):
    assert type(tgnlp_corpus) == Corpus, "Inputted data is not of type Corpus"

    sentence_corpus = tgnlp_corpus.sentence_corpus
    word_counts = tgnlp_corpus.word_counts

    #get syntactic pair edge list
    edge_dict = syntactic_pair_generation(sentence_corpus)

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
    assert type(tgnlp_corpus) == Corpus, "Inputted data is not of type Corpus"
    word_corpus = tgnlp_corpus.word_corpus
    sentence_corpus = tgnlp_corpus.sentence_corpus
    word_counts = tgnlp_corpus.word_counts
    word2vec_input = []
    for sentence in sentence_corpus:
        temp_word_list = []
        sentence_list = sentence.split(' ')
        #print(sentence_list)
        for word in sentence_list:
            temp_word_list.append(word)
        word2vec_input.append(temp_word_list)

    #using skipgram, better for small corpus
    model = gensim.models.Word2Vec(word2vec_input, min_count=1, vector_size=50, window=5, sg=1)

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
        similar = model.wv.most_similar(word1, topn = 50)

        #tuple of word and cosine similarity
        for word_similarity in similar:
            word2 = word_similarity[0]

            #word2vec can return empty string, need to make sure not to use
            if len(word2) == 0:
               continue
              
            #taking out normalization for semntic
            #norm_edgeweight = word_similarity[1]/(min(word_counts[word1], word_counts[word2]))
            sem_edgelist.append((word1, word2, word_similarity[1]))



    #TODO probably remove edgelist save
    with open('sem_edge_list.txt', 'w') as file:
        for edge in sem_edgelist:
            file.write(f"{edge[0]}, {edge[1]}, {edge[2]}\n")   
  
    return sem_edgelist


###########################################
####Sequential Graph Code##################
###########################################

#returns a networkx graph representing sequential relationships between words
#in the provided corpus
def get_sequential_graph(tgnlp_corpus, window_size=5):
  
  assert type(tgnlp_corpus) == Corpus, "Inputted data is not of type Corpus"  

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
    #create a weighted edge from word A to word B
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
    assert trim <= 1, "Provided value for trim is too large. Trim value must be <= 1"
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

   
