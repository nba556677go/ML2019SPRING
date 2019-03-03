import csv 
import numpy as np
import math
import sys
import random

def normalize(train):
    mean = np.mean(train , axis = 0)
    std = np.std(train, axis = 0)
    train -= mean
    train  = np.true_divide(train , std)
    print(mean.shape)
    print(std.shape)
    print(train)
    return train , mean , std

def feature_scaling(data):
    alldev = []
    allmean = []
    for i in range(len(data)):
        mean = sum(data[i]) / len(data[i])
        allmean.append(mean)
        dev = 0
        for j in range(len(data[i])):
            dev += (data[i][j] - mean)**2
        dev = np.sqrt(dev / (len(data[i]) - 1))
        alldev.append(dev)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]  = (data[i][j] - allmean[i]) / alldev[i]
    print(allmean)
    print(alldev)
    print(len(data[0]))
    input()

def read_val_index(valfile_x , features):
    val_index = []
    row = 0
    with open(valfile_x, 'r', encoding='big5') as f:
        lines  = f.readlines()
        for line in lines:
            if row % len(features) == 0:
                pos = line.find(',')
                val_index.append(int(line[3:pos]))
            row += 1
    #print(val_index , len(val_index))
    return val_index
data = []
for i in range(18):
	data.append([])


with open('data/train.csv', 'r', encoding='big5') as f:
    n_row = 0
    rows = csv.reader(f , delimiter=",")
    #print(row)
    for line in rows:
        if n_row != 0:
            for i in range(3,27):
                if line[i] != 'NR':
                    data[n_row%18-1].append(float(line[i]))
                else:
                    data[n_row%18-1].append(float(0))
        n_row += 1



train_x = []
train_y = []





#get NO2 , NOx , O3 , PM10 , PM2.5 , SO2
feature_list = [5,6,7,8,9,12]
#feature_list = [8,9]
#generate validation file
val_index = read_val_index('validation_x.csv' , feature_list)
ten_hr = 0
for each_hr in range(len(data[0])-9):#every row in data has equal length 5760
    if each_hr not in val_index:
        train_x.append([])
        for i in feature_list:
            for window in range(each_hr, each_hr+9):
                train_x[-1].append(data[i][window])
        train_y.append(data[9][each_hr+9])#last of every 10




train_x= np.array(train_x)
train_y = np.array(train_y)
#normalize
#train_x ,mean_x , std_x = normalize(train_x)
#mean_y = np.mean(train_y)
#std_y = np.std(train_y)
#print(mean_y)
#print(std_y)
#input()
#concatanate x**2
train_x = np.concatenate((train_x , train_x**2) , axis=1)
#print(train_x.shape)
#add bias
train_x = np.concatenate( (np.ones((train_x.shape[0] , 1)) , train_x), axis = 1)
print(train_x.shape)








#weight initialize
w = np.zeros(train_x.shape[1])
print(w.shape)
#training

x_t = train_x.transpose()

#############adagrad optimizer####################
l_rate = 1
iteration = 100000
s_grad = np.zeros(train_x.shape[1])
#for continue training...###############
#w = np.load('model.npy')
#s_grad = np.load('sgrad.npy')
########################################



for i in range(iteration):
    hypo = np.dot(train_x, w)
    loss = hypo - train_y
    cost = np.sum(loss**2) / len(train_x)
    cost_sq = math.sqrt(cost)
    grad = 2*np.dot(x_t , loss)
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - l_rate * grad/ada
    if (i%500==0):
        print("iteration: " , i , "cost: " , cost_sq)
    if (i%20000 == 0 ):
        np.save('model'+ str(i) +'.npy' , w)
#print(w)


"""
#################adam optimizer##################
b1 = 0.9#exponentialdecay rate for moment
b2 = 0.99
l_rate = 0.0001
epsilon  = 1e-8
iteration = 200000
#for continue training...###############
#w = np.load('model.npy')
#m = np.load('m.npy')
#v = np.load('v.npy')
########################################

#inintialize mass , velocity
m = np.zeros(train_x.shape[1])
v = np.zeros(train_x.shape[1])
for i in range(iteration):
    hypo = np.dot(train_x, w)
    loss = hypo - train_y
    cost = np.sum(loss**2) / len(train_x)
    cost_sq = math.sqrt(cost)
    grad = 2*np.dot(x_t , loss)
    m = b1*m + (1-b1)*grad
    v = b2*v + (1-b2)*np.multiply(grad,grad)
    m_hat = np.true_divide(m, (1-b1))
    v_hat = np.true_divide(v ,(1-b2))
    w = w - l_rate * m_hat / (np.sqrt(v_hat)+ epsilon)
    if (i%500==0):
        print("iteration: " , i , "cost: " , cost_sq )

    if (i%20000 == 0 ):
        np.save('model'+ str(i) +'.npy' , w)


"""
np.save('model.npy' , w)
#np.save('m.npy' , m)
#np.save('v.npy' , v)
np.save('sgrad.npy' , s_grad)
















            
   
        






    


