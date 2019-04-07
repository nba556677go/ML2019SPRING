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
from model import CNN
from trainer import trainer


def main():
# Hyper-parameters 
    input_size = 103
    num_classes = 7
    num_epochs = 150
    batch_size = 32
    validation_split = 0.1
    final_model_name = 'nn-epoch'+str(num_epochs)+'.pt'
    can_train = True
    is_validation = 1
    earlystop = 15
    ensemble=5
#########################################feature transforming###################################
    train_transform = transforms.Compose([  
                                    transforms.ToPILImage()  ,          
                                   # transforms.Resize((70, 70)),
                                   # transforms.RandomCrop((48, 48)),
                                    transforms.RandomHorizontalFlip(),
                                    #transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(22),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.5,), std = (0.5,)),
                                    ])
    val_transform = transforms.Compose([
            transforms.ToPILImage()  , 
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,), std = (0.5, )),
    ])
    test_transform = transforms.Compose([
           transforms.ToPILImage()  , 
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,), std = (0.5, )),
    ])
###################################### loading and cutting validation set########################
    train_dataset = MyDataset(  train_file =sys.argv[1] ,save= True, is_train = True, transform= train_transform )
  #  train_dataset = MyDataset(is_train = True,
   #                         loadfiles=("train_x.npy" , "train_y.npy") , transform= train_transform )
    #val_dataset = MyDataset(is_train = True,
     #                       loadfiles=("train_x.npy" , "train_y.npy") , transform= val_transform )    
  #  train_dataset = MyDataset(is_train = True,
   #                         save= True , transform= train_transform )
    val_dataset = MyDataset(is_train = True,
                            loadfiles=("train_x.npy" , "train_y.npy") , transform= val_transform )    
    
    if can_train == False:
        test_dataset = MyDataset(testx_file= sys.argv[2] , 
                            is_train = False , save = False , transform= test_transform)
        test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False) 
    
    if is_validation == 0:
        train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True)
    else:
        #creating validation set
        dataset_len = len(train_dataset)
        print(dataset_len)
        indices = list(range(dataset_len))
        val_len = int(np.floor(validation_split * dataset_len))
        validation_idx = np.random.choice(indices, size=val_len, replace=False)
        train_idx = list(set(indices) - set(validation_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)
        print("len train:" , len(train_idx))
        print("len valid:" , len(validation_idx))
        #print(train_idx)
        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                             sampler = train_sampler)
        validation_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            sampler = validation_sampler)

        for _, batch in  enumerate(train_loader):
            
            data_x , target = batch
            print ('Size of image:', data_x.size())  # batch_size*1*48*48
            print ('Type of image:', data_x.dtype)   # float32
            print ('Size of target:', target.size()) 
            break
####################################training##############################
    cnn = CNN()
    if can_train ==True:
        for i in range(ensemble):
            cnn = CNN()
            if is_validation:
                CNNtrainer = trainer(model = cnn , train_dataloader=train_loader
                        ,validation_loader=validation_loader)
            else:
                CNNtrainer = trainer(model = cnn , train_dataloader=train_loader)
            print("ensembling.." , i)
            CNNtrainer.train(num_epochs = num_epochs ,  is_validation = is_validation , max_earlystop= earlystop , ensemble = i)
    else :
       cnn = torch.load(sys.argv[3])
       CNNtrainer = trainer(model = cnn , test_dataloader=test_loader) 
       CNNtrainer.test()
if __name__ == "__main__":
    main()
    
