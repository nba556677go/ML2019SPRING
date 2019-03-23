import csv 
import numpy as np
import math
import sys
from logisticmodel import logisticmodel

def get_feature1(train_x):
    features = [0,2,3,4,5,61,62,63]
    train_x = train_x[: , features]
    
    return train_x

def get_feature2(train_x):
    #square
    continuous_feature = [0,1,3,5]
    #features = [0,2,3,4,5,61,62,63]
    square = train_x[: , continuous_feature]
    train_x = np.concatenate((train_x , square**2) , axis=1)
    return train_x 
def get_feature2_cha(train_x):
    #square
    continuous_feature = [0,3]
    #features = [0,2,3,4,5,61,62,63]
    square = train_x[: , continuous_feature]
    train_x = np.concatenate((train_x , square**2) , axis=1)
    return train_x 
def get_feature6(train_x):
    select = train_x[:, [0,3,63]]
    train_x = np.concatenate((train_x , select**2) , axis=1)
    #delete '?' column
    train_x = np.delete(train_x , [14,52,105] , 1)
    return train_x
def loaddata( trainx_file, trainy_file , normalization = 1):
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
                print(line)
                for i in range(len(line)):
                    if line[i] == ' White':
                        print(" White" , i)
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
    
    print(train_y.shape)
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

    train_x = get_feature6(train_x)
    #print(train_x)
    print(train_x.shape)
    #input()
    return train_x, train_y


    
def test(test_file , w , output_file, normalization = 1):
    test_x = []
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
        mean = np.mean(test_x, axis = 0) 
        #print(mean.shape)
        std = np.std(test_x, axis = 0)    
        for i in range(test_x.shape[0]):
            for j in continuous_feature:
                if not std[j] == 0 :
                    test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]         


    modeltmp = logisticmodel()
    test_x = get_feature6(test_x)
    test_x = modeltmp.add_bias(test_x)  
    predict = modeltmp.sigmoid(np.dot(test_x , w))
    with open(output_file , 'w') as f:
        subwriter = csv.writer(f , delimiter = ',')
        subwriter.writerow(["id" , "label"])
        for i in range(len(predict)):
            if predict[i] < 0.5 :
                subwriter.writerow([str(i+1) , 0])
            else:
                subwriter.writerow([str(i+1) , 1])
    return test_x

    
if __name__ == "__main__":
    #train = 1
    train = 0
    if train:
        train_x , train_y = loaddata(sys.argv[1] , sys.argv[2] , normalization= 1 )
        model = logisticmodel()
        w = model.train(train_x, train_y , epochs= 50000 , regularize = 0.01)
    #
    # test
    else:
        w = np.load("logistic-feature6-50000-regularize-0.01-0.85773.npy")
        test(sys.argv[3] , w , sys.argv[4] , normalization=1)
    
    
    
    
