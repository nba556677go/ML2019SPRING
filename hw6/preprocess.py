# standard library
import os
import csv
import sys
import argparse
from multiprocessing import Pool

# optional library
import jieba
import pandas as pd
from gensim.models import Word2Vec
import pickle
# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
class Preprocess():
    def __init__(self, data_dir, label_dir, args , loadtoken=False , test=False ):
        # Load jieba library
        jieba.set_dictionary(args.jieba_lib)
        self.test = test
        self.embed_dim = args.word_dim
        self.seq_len = args.seq_len
        self.wndw_size = args.wndw
        self.word_cnt = args.cnt
        self.save_name = 'word2vec'
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.sentence_count = 0
        if test == False:
            self.testx = args.testx
        # Load corpus
        
        if data_dir!=None:
            if loadtoken == False:
                # Read data
                dm = pd.read_csv(data_dir)
                data = dm['comment']
                #print(data)
                #input()
                if len(data) > 119018:
                    data = data[:119018]
                # Tokenize with multiprocessing
                # List in list out with same order
                # Multiple workers
                Pr = Pool(4) 
                data = Pr.map(self.tokenize, data)
                Pr.close()
                Pr.join()
                self.data = data
                if len(self.data) > 25000:#train
                    with open("token.pkl", 'wb') as f:
                        pickle.dump(self.data, f)
            else:
                with open("token.pkl", 'rb') as f:
                    self.data = pickle.load(f)
                    print(len(self.data))
                
            
        if label_dir!=None:
            # Read Label
            dm = pd.read_csv(label_dir)
            self.label = [int(i) for i in dm['label'][:119018]]
            print(len(self.label))

    def tokenize(self, sentence):
        # tokenize one sentence
        print('=== count {}'.format(self.sentence_count), end="\r")
        tokens = jieba.lcut(sentence)
        tokens = [i for i in tokens if (i != ' ') and (i != '')]
        self.sentence_count += 1
        return tokens

    def get_embedding(self, load=False):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load(self.save_name)
        else:
            #word2vec with testing vector
            if self.test==False:
                td = pd.read_csv(self.testx)
                test_data = td['comment']
                #print(type(test_data))
                # Multiple workers
                P = Pool(4) 
                test_data = P.map(self.tokenize, test_data)
                P.close()
                P.join()
        
               # print(test_data)
                #input()
                w2vdata = self.data+ test_data
                #print(len(self.data))
                #input()
            embed = Word2Vec(w2vdata, size=self.embed_dim, window=self.wndw_size, min_count=self.word_cnt, iter=16, workers=8)
            embed.save(self.save_name)
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        for i, word in enumerate(embed.wv.vocab):
            print('=== get words #{}'.format(i+1), end='\r')
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        self.vectors = torch.tensor(self.vectors)
        # Add special tokens
        self.add_embedding(self.pad)
        self.add_embedding(self.unk)
        print("=== total words: {}".format(len(self.vectors)))
        return self.vectors

    def add_embedding(self, word):
        # Add random uniform vector
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        if word == self.pad:
            vector = torch.zeros(1, self.embed_dim)
        #print(vector,word)
        #input()
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        #print(vector, word)
        #input()
        self.vectors = torch.cat([self.vectors, vector], 0)

    def get_indices_data(self,test=False):
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
        indices = []
        sentence_true_len = []
        # Use tokenized data
        for i, sentence in enumerate(self.data):
            print('=== get indices row #{}'.format(i+1), end='\r')
            
            sentence_true_len.append(len(sentence))
            #print(sentence_true_len[-1])
            #input()
            sentence_indices = []
            for word in sentence:
                if word in self.word2index:
                    # if word in dictionary give word vector indices
                    sentence_indices.append(self.word2index[word])
                else:  
                    # if not in dictionary give <unknown> vector indices
                    sentence_indices.append(self.word2index[self.unk])
            # pad all sentence to fixed length
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            indices.append(sentence_indices)
        if test:
            return torch.LongTensor(indices) , torch.LongTensor(sentence_true_len)         
        else:
            return torch.LongTensor(indices), torch.LongTensor(self.label)   ,torch.LongTensor(sentence_true_len)      

    def pad_to_len(self, arr, padded_len, padding=0):
     
        
        padded_len = int(padded_len)
        if (len(arr) < padded_len):
            for i in range(padded_len-len(arr)):
                arr.append(padding)
        return arr[:padded_len]