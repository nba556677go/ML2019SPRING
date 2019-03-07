


import numpy as np
import math
import sys
import csv

testing_data_path = sys.argv[1]
output_file_path  = sys.argv[2]


w = np.load('model-best.npy')
test_x = []
count  = 0
feature_list = [5,6,7,8,9,12]
#feature_list = [i for i in range(18)]
#feature_list =[3,4,5,6,7,8,9,10,12,13]
#feature_list = ["NMHC", "NO" ,"NO2" , "NOx" , "O3" ,"RAINFALL" ,"PM10" , "PM2.5" , "SO2"]
#feature_list = ["PM2.5"]
with open(testing_data_path , 'r') as f:
    lines  = f.readlines()
    for line in lines :
        line = line.rstrip('\n').split(',')
        #print(line)
        #add bias
        
        if count % 18 ==0 :
            test_x.append([])
        if count % 18 in feature_list:
            for i in range(2 , 11):
                if line[i] != 'NR':
                    test_x[count//18].append(float(line[i]))
                else :
                    test_x[count//18].append(float(0))
        count += 1

#print(test_x)
test_x = np.array(test_x)
#print(test_x.shape)
#test_x = feature1(test_x , feature_list)
#concatanate x**2
test_x = np.concatenate((test_x , test_x**2) , axis=1)
#add bias
test_x = np.concatenate( (np.ones((test_x.shape[0] , 1)) , test_x), axis = 1)

ans = []

predict = np.dot(test_x , w)

for i in range(len(predict)):
    ans.append(["id_"+ str(i), predict[i] ] )
with open(output_file_path , "w") as f:
    subwriter = csv.writer(f , delimiter = ',')
    subwriter.writerow(["id" , "value"])
    for i in range(len(ans)):
        subwriter.writerow(ans[i])
    

#print(test_x.shape)