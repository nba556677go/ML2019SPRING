import csv 
import numpy as np
import math
import sys

def feature_extract_1(data , month , date):#wind
    feat1 = np.multiply(data[16,month*480+date:month*480+date+9] , np.sin(data[15,month*480+date:month*480+date+9]*np.pi/180.))
    
    data1 = np.delete(data , [4,15,16,17] , 0)
    return np.concatenate((data1.flatten() , feat1.flatten() ) , axis = 0)


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
    print(data)
    input()
        

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
                if line[i] == 'NR' :
                    data[n_row%18-1].append(float(0))
                else:
                    data[n_row%18-1].append(float(line[i]))
        n_row += 1

#feature_scaling(data)

data = np.array(data)
    #get NO2 , NOx , O3 , PM10 , PM2.5 , SO2
feature_list = [5,6,7,8,9,12]
#feature_list = [0,1,2,3,5,6,7,8,9,10,11,12,13,14]
#feature_list = [i for i in range(18)]
def get_feature(data, feature_list):
    train_x = []
    train_y = []


    for month in range(12):#every row in data has equal length 5760
        for date in range(471):
            kill = False
            for i in range(18):
                if kill == True:
                    break
                for j in range(date , date+9):
                    if data[i][month*480+j] < 0:
                        kill = True
                        break
            if kill == False :
                #train = feature_extract_1(data , month ,date)
                train = data[feature_list,month*480+date:month*480+date+9] 
                train_x.append(train.flatten())
                train_y.append(data[9][month*480+date+9])#last of every 10

    
    train_x= np.array(train_x)
    train_y = np.array(train_y)



    #concataknate x**2
    train_x = np.concatenate((train_x , train_x**2) , axis=1)
    #print(train_x.shape)
    #add bias
    print(train_x.shape)
    train_x = np.concatenate( (np.ones((train_x.shape[0] , 1)) , train_x), axis = 1)
    print(train_x.shape)
    #input()
    print(train_y.shape)
    input()
    return train_x , train_y
    
train_x , train_y = get_feature(data , feature_list)
#weight initialize
w = np.zeros(train_x.shape[1])
l_rate = 1
iteration = 1500000

#training
x_t = train_x.transpose()
s_grad = np.zeros(train_x.shape[1])
#for continue training...###############
#w = np.load('model.npy')
#s_grad = np.load('sgrad.npy')
########################################
 #adagrad for regularization
#train_loss = [] 
#for lambdda in [0 , 0.1, 0.01 , 0.001 , 0.0001]:
#train_loss.append([])


for i in range(iteration):
    hypo = np.dot(train_x, w)
    #print("hypo :" , hypo.shape)
    loss = hypo - train_y
    #regular = np.multiply(w,w)
 #   regular = np.sum(w**2)
#    loss = loss + lambdda*0.5*regular
    cost = np.sum(loss**2) / len(train_x)
    cost_sq = math.sqrt(cost)
    grad = 2*np.dot(x_t , loss)
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - l_rate * grad/ada
    if (i%500==0):
        print("iteration: " , i , "cost: " , cost_sq)
    if (i%20000 == 0 ):
        
        np.save('model-6feature'+ str(i) +'.npy' , w)
        #train_loss[-1].append(cost_sq)
        
#np.save('model-regular-PM2.5'+ str(lambdda) +'.npy' , w)

#print(train_loss)

#################adam optimizer##################
"""
b1 = 0.9#exponentialdecay rate for moment
b2 = 0.999
b1_t = 1
b2_t = 1
l_rate = 0.01
epsilon  = 1e-8
iteration = 2000000
#inintialize mass , velocity
m = np.zeros(train_x.shape[1])
v = np.zeros(train_x.shape[1])

for i in range(iteration):
    hypo = np.dot(train_x, w)
    loss = hypo - train_y
    cost = np.sum(loss**2) / len(train_x)
    cost_sq = math.sqrt(cost)
    grad = 2*np.dot(x_t , loss)
    #print(grad.shape)
    b1_t *= b1
    b2_t *= b2
    m = b1*m + (1-b1)*grad
    v = b2*v + (1-b2)*np.dot(grad,grad)
    m_hat = np.true_divide(m, 1 - b1_t)
    v_hat = np.true_divide(v ,1 - b2_t)
    w = w - l_rate * m_hat / (np.sqrt(v_hat)+ epsilon)
    if (i%500==0):
        print("iteration: " , i , "cost: " , cost_sq )

    if (i%20000 == 0 ):
        np.save('model'+ str(i) +'.npy' , w)


#print(w)
"""
np.save('model-sixfeature-'+str(iteration)+'.npy' , w)
#np.save('m.npy' , m)
#np.save('v.npy' , v)
np.save('sgrad.npy' , s_grad)














            
   
        






    


