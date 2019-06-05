import csv 
import numpy as np
import math
import sys
from torch_data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from model import *
from trainer import trainer


def main():
# Hyper-parameters 
    input_size = 103
    num_classes = 7
    num_epochs = 261
    batch_size = 64
    validation_split = 0.1
    final_model_name = 'nn-epoch'+str(num_epochs)+'.pt'
    can_train = True
    is_validation = 1
    earlystop = 220
    ensemble=1
    out_path = ''
#########################################feature transforming###################################
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(54),
        transforms.RandomCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),
    ])
    val_transform = transforms.Compose([
            transforms.ToPILImage()  , 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5]),
    ])
###################################### loading and cutting validation set########################
    train_dataset = MyDataset(  train_file =sys.argv[1] ,save= True, is_train = True, transform= train_transform )
    val_dataset = MyDataset(is_train = True,
                            loadfiles=("train_x.npy" , "train_y.npy") , transform= val_transform )    
    #creating validation set
    dataset_len = len(train_dataset)
    indices = list(range(dataset_len))
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                            sampler = train_sampler)
    validation_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        sampler = validation_sampler)
####################################training##############################
    cnn = MobileNet_Li29()
    CNNtrainer = trainer(model = cnn , train_dataloader=train_loader
                ,validation_loader=validation_loader, prefix = 'nn')
    CNNtrainer.train(num_epochs = num_epochs ,  is_validation = is_validation , max_earlystop= earlystop , ensemble = 0)
if __name__ == "__main__":
    main()
    
