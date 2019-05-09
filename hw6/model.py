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

# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch.nn.functional as F
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(LSTM_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x dimension(batch, seq_len, hidden_size)
        getmax = F.adaptive_max_pool1d(x.permute(0,2,1), 1).squeeze()
        getavg = F.adaptive_avg_pool1d(x.permute(0,2,1), 1).squeeze()
        x = torch.cat((getmax, getavg) , 1)
        x = self.classifier(x)
        return x

class LSTM_Bi_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(LSTM_Bi_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True , bidirectional = True , dropout = 0.5)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*4, 256),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        #bi_h0 = torch.zeros(self.num_layers*2, inputs.size(0) ,self.hidden_dim).to(self.device) 
        #bi_c0 = torch.zeros(self.num_layers*2,inputs.size(0) , self.hidden_dim).to(self.device) #(batch , num_layers * num_directions, hidden_size)
       
        x, _ = self.lstm(inputs, None)
        getmax = F.adaptive_max_pool1d(x.permute(0,2,1), 1).squeeze()
        getavg = F.adaptive_avg_pool1d(x.permute(0,2,1), 1).squeeze()
        x = torch.cat((getmax, getavg) , 1)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state
        x = torch.cat((getmax, getavg) , 1)
        x = self.classifier(x)
        return x


class GRU_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(GRU_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.gru1(inputs, None)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

class Mix_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(Mix_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = dropout
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers,inputs.size(0) , self.hidden_dim).to(self.device)
        c1 = torch.zeros(self.num_layers,inputs.size(0) , self.hidden_dim).to(self.device) #(batch , num_layers * num_directions, hidden_size)
        inputs = self.embedding(inputs)
    
       
        """
        #print(inputs.size(0))
        #print(inputs.size(1))
        true_len=[]
        for i in range(inputs.size(0)):
            for j in range(inputs.size(1)):
                #print(inputs[i,j,:])
                if inputs[i,j,:].float().sum() == 0:
                    #print(inputs[i,j,:])
                    true_len.append(j)
                    break
        print(true_len)
        input()

        #inputs[:-1:]
        #print(inputs.size())
        #input()
        """
        x, h1 = self.gru1(inputs, h0)
        x , _= self.lstm1(x , (h1 , c1) )
  
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state
        getmax = F.adaptive_max_pool1d(x.permute(0,2,1), 1).squeeze()
        getavg = F.adaptive_avg_pool1d(x.permute(0,2,1), 1).squeeze()
        #getmax = torch.max(x,dim=1, keepdim=True)[0]
        #getavg = torch.mean(x,dim=1, keepdim=True)
        #print(x.size())
        #print(getmax.size())
        #print(getavg.size())
        x = torch.cat((getmax, getavg) , 1)
        #print(x.size())
        #input()
       # x = x[:, -1, :]
        x = self.classifier(x)
        return x
class Mix2_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(Mix2_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = dropout
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout = 0.3)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        #c1 = torch.zeros(self.num_layers,inputs.size(0) , self.hidden_dim).to(self.device) #(batch , num_layers * num_directions, hidden_size)
        inputs = self.embedding(inputs)
        x, (h1,c1) = self.lstm1(inputs, None)
        x , _= self.gru1(x , h1 )
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


class Mix_Bi_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(Mix_Bi_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = dropout
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout = 0.3)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True , bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Dropout(dropout), 
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),

            nn.Sigmoid())
    def forward(self, inputs):
        c1 = torch.zeros(self.num_layers*2,inputs.size(0) , self.hidden_dim).to(self.device) #(batch , num_layers * num_directions, hidden_size)
        
        inputs = self.embedding(inputs)
        x, h1 = self.gru1(inputs, None)
        #x = torch.cat((x[:, -1, :self.hidden_dim] , x[:,0 , self.hidden_dim:]) , 1)
        #print(x.size())
        h1 = torch.cat((h1,h1) , 0 )
        #print(h1.size())
        x , _= self.lstm1(x , (h1 , c1) )
        x = torch.cat((x[:, -1, :self.hidden_dim] , x[:,0 , self.hidden_dim:]) , 1)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state
      
       # print(x.size())
        x = self.classifier(x)
        return x