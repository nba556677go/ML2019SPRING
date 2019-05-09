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
import numpy as np
# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct
def training(args, train, valid, model, device, ensemble_num):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    model.train()
    batch_size, n_epoch = args.batch, args.epoch
    criterion = nn.BCELoss()
    t_batch = len(train) 
    v_batch = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_loss, total_acc, best_acc , stop_cnt = 0, 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training set
        for i, (inputs, labels) in enumerate(train):
            #input(type(inputs))
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)

            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{} == {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f} '.format(total_loss/t_batch, total_acc/t_batch*100))
        if stop_cnt > args.earlystop:
            print(f"earlystopping since stop count have reached {args.earlystop}")
            torch.save(model, "{}/ensemble{}lstm_shieh_{:.3f}_seq{}_dim{}_numlayer{}_epoch{}".format(args.model_dir,ensemble_num,total_acc/v_batch*100, args.seq_len,args.hidden_dim,args.num_layers, epoch))
            print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
            break
        # validation set
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                 stop_cnt = 0
                 best_acc = total_acc
                 torch.save(model, "{}/ensemble{}lstm_shieh_{:.3f}_seq{}_dim{}_numlayer{}_epoch{}".format(args.model_dir,ensemble_num,total_acc/v_batch*100, args.seq_len,args.hidden_dim,args.num_layers, epoch))
                 print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
            else :
                stop_cnt += 1
        model.train()
def ensemble(model,test_loader, device ):
    model.eval()
    predict_list = []
    
        #total_loss, total_acc = 0, 0
   
    for i,data in enumerate(test_loader):
        #data = torch.LongTensor(data)
        #for i in range(1 ,len(data)):
        #    assert(data[i].size() == data[i-1].size())
        data = torch.stack(data , dim= len(data))
        #print(data)
        #print(data.size())
        data = data.view(len(data) , -1)
        #print(data.size())
        #print(data)
        data = data.to(device, dtype=torch.long)
        

        #labels = labels.to(device, dtype=torch.float)
        outputs = model(data)
        
        outputs = outputs.squeeze()
        #outputs[outputs>=0.5] = 1
        #outputs[outputs<0.5] = 0
        #print(outputs.size())
        #input()
        #input(outputs)
        outputs = outputs.cpu().detach().numpy()
        predict_list.append(outputs)
        
    #print(len(predict_list))
    #print(len(predict_list[0]))
    #input()
    return np.concatenate(predict_list , axis = 0)

def testing(args, test_loader, device):
    model_list = os.listdir(args.model_dir)
    predict_list = []
    print(model_list)
    for  i in model_list:
        print(f"ensembling model {i}...")
        i = os.path.join(args.model_dir , i)
        model = torch.load(i)
        model.to(device)
        predict_list.append(ensemble(model = model, test_loader = test_loader , device = device))
        #print(predict_list[-1].shape)
    #print(predict_list)
    # predict_list = np.array(predict_list).squeeze()
    #print(predict_list.shape)
    #input()
    #mean

    #if args.ensemble !=1:
    #    predict_list = predict_list.mean(axis = 0)
    #    predicts = predict_list.argmax(axis = 1)
    #else:
        #predicts = predict_list.squeeze()
    #print(len(predict_list))
    #predicts= np.concatenate(predict_list , axis = 0)
    predicts = np.array(predict_list)
    print(predicts.shape)
    predicts = predicts.mean(axis = 0)
    
    print(predicts)
    print(predicts.shape)
    #predicts = predicts.argmax(axis = 1)
    predicts[predicts>=0.5] = 1
    predicts[predicts<0.5] = 0
    print(predicts)
    #input()
    writepredict(predict = predicts , output = args.output)

def writepredict(predict , output):
    #flat_list = [item for sublist in predict for item in sublist]
    with open(output , 'w') as f:
        subwriter = csv.writer(f , delimiter = ',')
        subwriter.writerow(["id" , "label"])
        for i in range(len(predict)):
            subwriter.writerow([str(i) , int(predict[i])])
