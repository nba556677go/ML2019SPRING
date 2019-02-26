import csv 
import numpy as np
import math
import sys

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
ten_hr = 0
for each_hr in range(len(data[0])-9):#every row in data has equal length 5760
    #ten_hr += 
    #if ten_hr % 10 == 0:
    train_x.append([])
    for i in feature_list:
        for window in range(each_hr, each_hr+9):
            train_x[-1].append(data[i][window])
    train_y.append(data[9][each_hr+9])#last of every 10

    #elif each_hr%10 == 0:# create empty set every 9 hr
     #   train_x.append([])
      #  for next_nine in range(each_hr , each_hr + 9):
       #     for i in range(18):
        #        train_x[-1].append(data[i][next_nine])

train_x= np.array(train_x)
train_y = np.array(train_y)



#concataknate x**2
train_x = np.concatenate((train_x , train_x**2) , axis=1)
#print(train_x.shape)
#add bias
train_x = np.concatenate( (np.ones((train_x.shape[0] , 1)) , train_x), axis = 1)
print(train_x.shape)
print(train_y.shape)

#weight initialize
w = np.zeros(train_x.shape[1])
print(w.shape)
l_rate = 0.01
iteration = 200000

#training
x_t = train_x.transpose()
#s_grad = np.zeros(train_x.shape[1])
#for continue training...###############
w = np.load('model.npy')
s_grad = np.load('sgrad.npy')
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
np.save('model-5feature.npy' , w)
np.save('sgrad-5feature.npy' , s_grad)














            
   
        






    


