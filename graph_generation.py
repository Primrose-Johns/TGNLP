import os
import pickle
import string
import pandas as pd
from corenlp import StanfordCoreNLP
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


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize



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


#preprocess data for syntactic graph
def syn_corpus_preprocessing(data):
    #generate list of sentences
    sentence_corpus = data_to_corpus(data, 'sentence')

    #preprocess data to one large string
    word_corpus  = data_to_corpus(data, 'word')

    word_counts = get_word_counts(word_corpus)

    return sentence_corpus, word_counts


#creates the graph from preprocessed data
def get_syntactic_graph(sentence_corpus, word_counts):
    
    #get syntactic pair edge list
    edge_dict = syntactic_pair_generation(sentence_corpus)

    return syn_graph_generation(edge_dict)





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


def syn_graph_generation(edge_dict):
    #create graph
    G = nx.Graph()
    for (node1, node2), weight in edge_dict.items():
      G.add_edge(node1, node2, weight=weight)
    return G

            
           



###########################################
####Semantic Graph Code####################
###########################################


def sem_corpus_preprocessing(data):
    #list of sentences
    sentence_corpus = data_to_corpus(data, 'sentence')

    #list of words
    word_corpus = data_to_corpus(data, 'word')
    word_counts = get_word_counts(word_corpus)

    return sentence_corpus, word_corpus, word_counts


def get_semantic_graph(word_corpus, sentence_corpus, word_counts):
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

            norm_edgeweight = word_similarity[1]/(min(word_counts[word1], word_counts[word2]))
            sem_edgelist.append((word1, word2, norm_edgeweight))



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
def get_sequential_graph(corpus, word_counts, window_size=5):
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
####Normaliztion Code######################
###########################################
def trim_norm_graph(G_full, trim = 0.1):
    G = deepcopy(G_full)    
    #scaling portion, this will normalize the values between 0 and 1
    #we have to rerun these now that edges have been removed
    weights = [e[2]["weight"] for e in list(G.edges(data=True))]
    weights = np.array(weights).reshape(-1, 1)
    edges = sorted(G.edges(data=True), key=lambda val : val[2]["weight"])
    scaler = MinMaxScaler()
    scaler.fit(weights)
    for a, b, d in edges:
        weight = np.array(d["weight"]).reshape(-1, 1)
        new_weight = scaler.transform(weight)
        G[a][b]["weight"] = new_weight[0][0]

    #Triming portion
    #this just gets us a list of weights
    weights = [e[2]["weight"] for e in list(G.edges(data=True))]
    #print(weights[:10])
    #simple integral (no width of rectangle needs to be considered)
    auc = sum(weights)
    trim_amount = auc*trim
    trimmed = 0
    #get the edges in ascending order of weight
    edges = sorted(G.edges(data=True), key=lambda val : val[2]["weight"])
    #remove nodes until trim_amount of weights have been removed
    for a, b, d in edges:
        weight = d["weight"]
        if trimmed+weight < trim_amount:
            G.remove_edge(a, b)
            trimmed += weight
        else:
            break
    return G

if __name__ == '__main__':
    #load in bbc dataset
    data = loadbbc()

    
    #sequential graph generation
    corpus = data_to_corpus(data, 'word')
    word_counts = get_word_counts(corpus)

    start = timeit.default_timer()
    G1 = get_sequential_graph(corpus, word_counts)

    stop = timeit.default_timer()
    print('Sequential graph generation time: ', stop - start)
    print(G1)
    



    #syntatic graph generation
    #generate list of sentences and word count dictionary
    sentence_corpus, word_counts = syn_corpus_preprocessing(data)

    start = timeit.default_timer()

    #generate networkd graph
    G2 = get_syntactic_graph(sentence_corpus, word_counts)

    stop = timeit.default_timer()
    print('Syntactic graph generation time: ', stop - start)

    print("syntactic graph", G2)

    


    
    #semantic graph generation
    start = timeit.default_timer()

    sentence_corpus, word_corpus, word_counts = sem_corpus_preprocessing(data)
    G3 = get_semantic_graph(word_corpus, sentence_corpus, word_counts)
    
    stop = timeit.default_timer()
    print('Semantic graph generation time: ', stop - start)

    print("semantic graph", G3)

    


 




    

    

    
    
   
