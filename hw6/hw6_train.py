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
from trainer import training , evaluation
from preprocess import Preprocess
from model import *
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpudevice
    preprocess = Preprocess(args.train_x, args.train_y, args, loadtoken=True)
    # Get word embedding vectors
    embedding = preprocess.get_embedding(load=True)
    # Get word indices
    data, label , seq_true_len= preprocess.get_indices_data()
    #print(seq_true_len[0])
    dataset = TensorDataset(data,label)
    for i in range(args.ensemble_start, args.ensemble):
        # Split train and validation set
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size 
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set,batch_size=args.batch)
        # Get model
        print(f"training model{i}...")
        model = LSTM_Bi_Net(embedding, args.word_dim, args.hidden_dim, args.num_layers , fix_emb=True)
        model = model.to(device)
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        # Start training
        training(args, train_loader, val_loader, model, device , i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='[Output] Your model checkpoint directory')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('train_x',type=str, help='[Input] Your train_x.csv')
    parser.add_argument('train_y',type=str, help='[Input] Your train_y.csv')

    parser.add_argument('--fraction', default=0.8, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--seq_len', default=50, type=int)
    parser.add_argument('--word_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--wndw', default=3, type=int)
    parser.add_argument('--cnt', default=3, type=int)
    parser.add_argument('--gpudevice', default="1", type=str)
    parser.add_argument('--earlystop', default=10, type=int)
    parser.add_argument('--testx', default="data/test_x.csv", type=str)
    parser.add_argument('--ensemble', default=1, type=int)
    parser.add_argument('--ensemble_start', default=0, type=int)
    args = parser.parse_args()
    main(args)
