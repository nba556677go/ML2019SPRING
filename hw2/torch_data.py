import torch
import torch.nn as nn
import csv 
import sys
from torch.utils.data import Dataset
import numpy as np

def find_feature(train_file= "X_train"):
    feature = {}
    with open(train_file, 'r' , encoding = 'utf-8') as f:
        n_row = 0
        rows = csv.reader(f , delimiter=",")
        for line in rows:
            if n_row!= 0:
                break
            else:
                for i in range(len(line)):
                    if line[i][0] == ' ':
                        line[i] = line[i][1:]
                    feature[line[i]] = i
            n_row+= 1
    #print(feature)
    return feature
def get_feature2(train_x):
    #square
    continuous_feature = [0,1,3,5]
    #features = [0,2,3,4,5,61,62,63]
    #train_x = np.delete(train_x , [14,52,105] , 1)
    square = train_x[: , continuous_feature]
    train_x = np.concatenate((train_x , square**2) , axis=1)
    return train_x 
def get_feature3(train_x):
    #clear ? features
    train_x = np.delete(train_x , [14,52,105] , 1)
    return train_x
def get_feature4(train_x):
    feature_dict = find_feature()
    top_feat = ["capital_gain" , "Never-married" , "Married-civ-spouse" , "Bachelors" , 
    "Own-child" , "Exec-managerial" , "Other-service" , "Unmarried" , "Not-in-family" , "Divorced"]
    top = []
    for feat in top_feat:
        if feat in feature_dict:
            top.append(feature_dict[feat])
    print(top)
    top.sort()
    train_x = train_x[: , top]
    return train_x
def get_feature5(train_x):
    
    feature_dict = find_feature()
    top_feat = ["capital_gain" , "Never-married" , "Married-civ-spouse" , "Bachelors" , 
    "Own-child" , "Exec-managerial" , "Other-service" , "Unmarried" , "Not-in-family" , "Divorced", "White"]
    top = []
    for feat in top_feat:
        if feat in feature_dict:
            top.append(feature_dict[feat])
    print(top)
    top.sort()
    train_x = np.concatenate((train_x , train_x[: , top]**2) , axis=1)
    train_x = np.delete(train_x , [14,52,105] , 1)#delete '?'
    #add bias
    train_x = np.concatenate( (np.ones((train_x.shape[0] , 1)) , train_x), axis = 1)
    #print(train_x.shape)
    return train_x

def get_feature6(train_x):
    select = train_x[:, [0,3,63]]
    train_x = np.concatenate((train_x , select**2) , axis=1)
    #delete '?' column
    train_x = np.delete(train_x , [14,52,105] , 1)
    return train_x
def get_feature7(train_x):
    train_x = train_x[:, [0,3,63]]
    train_x = np.concatenate((train_x , train_x**2) , axis=1)
    #add bias
    #train_x = np.concatenate( (np.ones((train_x.shape[0] , 1)) , train_x), axis = 1)
    return train_x
def get_feature8(train_x):
    select = [0,1,3,4,5]
    select_feat = train_x[:, select]
    train_x = np.concatenate((train_x , select_feat**2) , axis=1)
    return train_x
    
    #add bias
    #train_x = np.concatenate( (np.ones((train_x.shape[0] , 1)) , train_x), axis = 1)
def get_feature9(train_x):
    select = [0,2,3,4]
    select_feat = train_x[:, select]
    train_x = np.concatenate((train_x , select_feat**2) , axis=1)
    return train_x
       

def _normalize_column_0_1(X, train=True, specified_column = None, X_min = None, X_max=None):
    # The output of the function will make the specified column of the training data 
    # from 0 to 1
    # When processing testing data, we need to normalize by the value 
    # we used for processing training, so we must save the max value of the 
    # training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_max = np.reshape(np.max(X[:, specified_column], 0), (1, length))
        X_min = np.reshape(np.min(X[:, specified_column], 0), (1, length))
    print(X_max , X_min)
    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_min), np.subtract(X_max, X_min))
    
    return X, X_max, X_min

def loaddata( trainx_file, trainy_file , normalization = 1 , normalize_0_1 = 0):
    train_x, train_y = [], []
    with open(trainx_file, 'r' , encoding = 'utf-8') as f:
        n_row = 0
        rows = csv.reader(f , delimiter=",")
        #print(row)
        
        for line in rows:
            if n_row != 0:
                train_x.append([])
                for i in range(len(line)):
                    train_x[-1].append(float(line[i]))
            else:
              #  print(line)
                for i in range(len(line)):
                    if  '?' in line[i] :
                        pass
                        #print("?" , i)
            n_row += 1

    with open(trainy_file, 'r' , encoding = 'utf-8') as f:
        n_row = 0
        rows = csv.reader(f , delimiter=",")
        #print(row)
        for line in rows:
            if n_row != 0:
                for i in range(len(line)):
                    train_y.append(float(line[i]))
            n_row += 1
    #print(train_x)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    #print(train_y.shape)
    if normalization:
        #normalize continuous feature
        continuous_feature = [0,1,3,4,5]
        mean = np.mean(train_x, axis = 0) 
        #print(mean.shape)
        std = np.std(train_x, axis = 0)    
        for i in range(train_x.shape[0]):
            for j in continuous_feature:
                if not std[j] == 0 :
                    train_x[i][j] = (train_x[i][j]- mean[j]) / std[j]    

    if normalize_0_1:
        continuous_feature = [0,1,3,5]
        train_x , mean  , std = _normalize_column_0_1(train_x , 
                                                        specified_column=continuous_feature)  
    train_x = get_feature6(train_x)
    #print(train_x)
    feature_size = train_x.shape[1]
    print(train_x.shape)
    #input()
    #input()
    return train_x, train_y , feature_size ,  mean , std

def loadtest(test_file  , mean , std , normalization = 1 , normalize_0_1 = 0):
    test_x = []
    #print(mean.shape , std.shape)
    with open(test_file , 'r') as f:
        n_row = 0
        rows = csv.reader(f , delimiter=",")  
       
        for line in rows:
            if n_row != 0:
                test_x.append([])
                for i in range(len(line)):
                    test_x[-1].append(float(line[i]))
            n_row += 1

    test_x = np.array(test_x)

    if normalization:
        continuous_feature = [0,1,3,4,5]
        #mean = np.mean(test_x, axis = 0) 
        #print(mean.shape)
        #std = np.std(test_x, axis = 0)    
        for i in range(test_x.shape[0]):
            for j in continuous_feature:
                if not std[j] == 0 :
                    test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]         

    if normalize_0_1:
        continuous_feature = [0,1,3,4,5]
        test_x , _  , _ = _normalize_column_0_1(test_x ,train= False, 
                                                    specified_column=continuous_feature,
                                                    X_max= mean ,X_min= std)
    #modeltmp = logisticmodel()
    test_x = get_feature6(test_x)
    #test_x = modeltmp.add_bias(test_x)  
    #predict = modeltmp.sigmoid(np.dot(test_x , w))
    print(test_x.shape)
   
    return test_x

class MyDataset(Dataset):
    def __init__(self, trainx_file, trainy_file , testx_file = None , train = False , mean = None , std = None):
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
        self.is_train = train		
        if self.is_train == True:
            self.train_x , self.train_y , self.feature_size  , self.mean , self.std = loaddata(trainx_file , trainy_file , 
                                                                                normalization=1 , normalize_0_1 = 0)
       
        elif trainx_file == None and self.is_train == False:
            #for testing data
            self.test_x  = loadtest(test_file = testx_file , mean = mean , std = std,
                                        normalization=1 , normalize_0_1 = 0)
            
            
    def __len__(self):
        #print("len: " , self.train_y.shape[0])
        if self.is_train == True:
            return self.train_y.shape[0]
        else: 
            #print("len: " , )
            return self.test_x.shape[0]
    
    def __getitem__(self, idx):
        #, label = self.label[idx]
       	#
        # imread: a function that reads an image from path
        
        #img = imread(img_path)
        
        # some operations/transformations
        if self.is_train == True:
            return torch.tensor(self.train_x[idx]), torch.tensor(self.train_y[idx])
        else: #train == false
            return torch.tensor(self.test_x[idx])

#data = MyDataset(sys.argv[1] , sys.argv[2])
#print(len(data))