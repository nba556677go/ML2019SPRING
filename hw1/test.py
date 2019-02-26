
import numpy as np
import math
import sys
import csv

w = np.load('model-5feature.npy')
test_x = []
count  = 0
feature_list = ["NO2" , "NOx" , "O3" , "PM10" , "PM2.5" , "SO2"]
with open('data/test.csv' , 'r') as f:
    lines  = f.readlines()
    for line in lines :
        line = line.rstrip('\n').split(',')
        #print(line)
        #add bias
        
        if count % 18 ==0 :
            test_x.append([])
        if line[1] in feature_list:
            for i in range(2 , 11):
                if line[i] != 'NR':
                    test_x[count//18].append(float(line[i]))
                else :
                    test_x[count//18].append(float(0))

        count += 1

#print(test_x)
test_x = np.array(test_x)
print(test_x.shape)
#concatanate x**2
test_x = np.concatenate((test_x , test_x**2) , axis=1)
#add bias
test_x = np.concatenate( (np.ones((test_x.shape[0] , 1)) , test_x), axis = 1)

ans = []

predict = np.dot(test_x , w)

for i in range(len(predict)):
    ans.append(["id_"+ str(i), predict[i] ] )
with open("submission.csv" , "w") as f:
    subwriter = csv.writer(f , delimiter = ',')
    subwriter.writerow(["id" , "value"])
    for i in range(len(ans)):
        subwriter.writerow(ans[i])
    

print(test_x.shape)