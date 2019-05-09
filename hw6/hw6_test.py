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
from trainer import  testing
from preprocess import Preprocess
from model import LSTM_Net

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess(args.test_x, None, args , test=True)
    # Get word embedding vectors
    embedding = preprocess.get_embedding(load=True)
    # Get word indices
    data, _ = preprocess.get_indices_data(test=True)
    #print(type(data[0]))
    test_set = TensorDataset(data)

    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch, shuffle=False)
    #val_loader = torch.utils.data.DataLoader(val_set,batch_size=args.batch)
    # Get model
    #model = LSTM_Net(embedding, args.word_dim, args.hidden_dim, args.num_layers)
    #print(str(args.model))
    #model = torch.load(args.model)
    #model.to(device)

    # Start testing
    testing(args, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_x',type=str, help='[Input] Your test_x.csv')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('output',type=str, help='[Output] name of prediction csv')
    #parser.add_argument('train_y',type=str, help='[Input] Your train_y.csv')

    parser.add_argument('--fraction', default=0.8, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--seq_len', default=30, type=int)
    parser.add_argument('--word_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--wndw', default=3, type=int)
    parser.add_argument('--cnt', default=3, type=int)
    parser.add_argument('--ensemble', default=1, type=int)
    parser.add_argument('--model_dir', default="models", type=str)
    args = parser.parse_args()
    main(args)