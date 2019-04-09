import csv
import numpy as np
def main():
    class_index = [0,299,2,7,3,15,4]
    read_classindex= [i+1 for i in class_index]
    train_x = []
    with open('../hw3/train.csv', 'r' , encoding = 'utf-8') as f:
        n_row = 0
        rows = csv.reader(f , delimiter=",")
        #print(row)
        
        for line in rows:
            if n_row != 0:
                if n_row in read_classindex:
                #train_y.append(float(line[0]))
                    train_x.append([float(i) for i in line[1].split(" ")])
                # for i in range(1 , len(line)):
                #    train_x[-1].append(float(line[i]))
            n_row += 1

        #reshape

        #print(train_x)
    train_x = np.array(train_x)
    #train_x = train_x[class_index,:]
    print(train_x.shape)
    train_x = np.reshape(train_x , (train_x.shape[0]  , 48 , 48 ))
    print(train_x)
    print(train_x.shape)
    train_x = np.reshape(train_x , (train_x.shape[0] ,1 , 48 , 48 ))
    print(train_x)
    print(train_x.shape)
    np.save("p2.npy" , train_x)

def loadval(trainfile):
    val_len = 2870
    train_x = []
    train_y = []
    with open(trainfile, 'r' , encoding = 'utf-8') as f:
        n_row = 0
        rows = csv.reader(f , delimiter=",")
        #print(row)
        
        for line in rows:
            if n_row == 2871 :
                break
            if n_row != 0 :
                #if n_row in read_classindex:
                train_y.append(float(line[0]))
                train_x.append([float(i) for i in line[1].split(" ")])
                # for i in range(1 , len(line)):
                #    train_x[-1].append(float(line[i]))
            n_row += 1

        #reshape

        #print(train_x)
    train_x = np.array(train_x)
    #train_x = train_x[class_index,:]
    #print(train_x.shape)
    train_x = np.reshape(train_x , (train_x.shape[0]  , 48 , 48  ))
    #print(train_x)
    #print(train_x.shape)
    #train_x = np.reshape(train_x , (train_x.shape[0] ,1 , 48 , 48 ))
    #print(train_x)
    #print(train_x.shape)
    train_y = np.array(train_y, dtype = np.float)
    np.save("val_x.npy" , train_x)
    np.save("val_y.npy" , train_y)
    return train_x , train_y

if __name__ == '__main__':
    main()