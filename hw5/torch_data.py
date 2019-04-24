import torch
import torch.nn as nn
import csv 
import sys
from torch.utils.data import Dataset
import numpy as np
from  PIL import Image
import os
import pickle
def loaddata(is_train  ,save = True , train_file = None, label_file = None , random= False):
    if is_train:
        train_x, train_y = [], []
        datas = os.listdir(train_file)
        train_x = [os.path.join(train_file, i) for i in datas if i[-3:] == 'png' ]
        train_x.sort()
        if label_file != None:
            if  'train_y.npy' not in label_file:
                if random==False:
                    with open(label_file, 'r' , encoding = 'utf-8') as f:
                        n_row = 0
                        rows = csv.reader(f , delimiter=",")
                        #print(row)

                    
                        for line in rows:
                            if n_row != 0:
                                train_y.append(int(line[3]))
                            n_row += 1
                    
            else:
                train_y = np.load(label_file)
        #reshape
        elif random == True:
            train_y = np.random.randint(255 , size=200)
        #print(train_x)
        #train_x = np.array(train_x)
        #print(train_x)

#        print(train_x.shape)
        train_y = np.array(train_y)
       # train_y = np.reshape(train_y , (train_y.shape[0] , 1))
        #print(train_y)
        #input()
        #print(train_y.shape)
        #input()

   
        #train_x = get_feature6(train_x)
        #print(train_x)
       
        
        #input()
        #input()
        if label_file != None:
            #np.save("train_x.npy" , train_x)
            np.save("train_y.npy" , train_y)

        return train_x , train_y

    ##############################load testing set #################################
 

class hw5Dataset(Dataset):
    def __init__(self, is_train = True ,  random= False, loadfiles= False , save=False ,  train_file = None ,label_file = None ,  testx_file = None  ,mean = None , std = None , transform = None ):
        """
        let's assume the csv is as follows:
        ================================
        image_path                 label
        imgs/001.png               1     
        imgs/002.png               0     
        imgs/003.png               2     
        imgs/004.png               1     
                      .
                      .
                      .
        ================================
       	And we define a function parse_csv() that parses the csv into a list of tuples 
       	[('imgs/001.png', 1), ('imgs/002.png', 0)...]
        """
        self.is_train = is_train
        ##image transform
        self.transform = transform
        self.labelfile = label_file
        #load from file
        self.loadfiles= loadfiles
        if loadfiles != False:
            #print("in1")
            if self.is_train == True:
                
                with open(loadfiles, 'rb') as f:
                    self.imagelist = pickle.load(f)
                #self.train_y = np.load(loadfiles)
                #self.train_x , _  = loaddata(is_train = self.is_train ,save = save ,train_file= train_file)    

            #else :
            #    self.test_x = np.load(loadfiles)
                
        else:
            #print("in2")
            self.train_x , self.train_y  = loaddata(is_train = self.is_train ,save = save ,train_file= train_file , label_file=label_file, random=random )  


   
            
    def __len__(self):
        #print("len: " , self.train_y.shape[0])
        if self.loadfiles != False:
            return len(self.imagelist)
        elif self.is_train == True and self.labelfile != None :
            #print(self.train_y.shape[0])
            #input()
            return self.train_y.shape[0]
        elif self.is_train == True and self.labelfile == None :
            print(len(self.train_x))
            #input()
            return  len(self.train_x)
        else: 
            #print("len: " , )
            return self.test_x.shape[0]
    
    def __getitem__(self, idx):
        #, label = self.label[idx]
       	#
        # imread: a function that reads an image from path
        
        #img = imread(img_path)
        if self.loadfiles!= False:
            assert idx < self.__len__()
            print(idx)
            im_path, label = self.imagelist[idx]
            img = Image.open(im_path)
            label = np.array(label)
            if self.transform is not None:
                img = self.transform(img)
            #print(img)
            #print(torch.tensor(label))
            #input()
            return img, torch.tensor(label)
        # some operations/transformations
        elif self.is_train == True:
            #print(idx)
            x = Image.open(self.train_x[idx])

            if self.transform is not None:
                #self.train_x[idx] = self.train_x[idx].astype(np.uint8)
                #train_x[idx] = Image.fromarray(train_x[idx].astype('uint8'), 'RGB')
                aftertransform = self.transform(x)
            #    print(aftertransform)
                #print(aftertransform.size())
                #print(aftertransform.dtype)

                #print(aftertransform)
                #print(np.array(x).shape)
                #print(aftertransform.size())
                
                #input()
                #print(torch.LongTensor(np.array(self.train_y[idx])))
                #print(np.array(self.train_y[idx]))
                #print(torch.tensor(np.array(self.train_y[idx])))
                #input()
                if self.labelfile != None:
                    return aftertransform, torch.tensor(np.array(self.train_y[idx]))
                else:
                    return aftertransform ,torch.tensor(np.array(self.train_y[idx]))

            return x, torch.LongTensor(np.array(self.train_y[idx]) )
"""
        else: #for test data
            x = torch.Tensor(self.test_x[idx])
           # print(self.train_x[idx].shape)
            x =x.view(-1 , 48 , 48)
            if self.transform is not None:
                #self.train_x[idx] = self.train_x[idx].astype(np.uint8)
                aftertrans = self.transform(x)
                return aftertrans
            return x
"""
#data = MyDataset(sys.argv[1] , sys.argv[2])
#print(len(data))
