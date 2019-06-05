import csv 
import numpy as np
import math
import sys
from torch_data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
from trainer import trainer
from model import *
import torchvision.transforms as transforms
batch_size = 128
num_transform = 1
num_emsemble = 1
test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),])         

def main():
    predict_list = []
    for count in range(num_transform):
        if count == 0:
            test_dataset = MyDataset(testx_file= sys.argv[1] , 
                                    is_train = False , save = True , transform= test_transform)
        else:
            test_dataset = MyDataset(loadfiles= "test_x.npy" , 
                                    is_train = False , save = False , transform= test_transform)            
        test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False) 
        for i in range(num_emsemble):
            predict_list.append(ensemble(modelname = 'model.pt', test_loader = test_loader))
    predict_list = np.array(predict_list)
    predict_list = predict_list.mean(axis = 0)
    predicts = predict_list.argmax(axis = 1)
    writepredict(predict = predicts , output = sys.argv[2])
    

def ensemble(modelname,test_loader ):
    model = MobileNet_Li29()
    if torch.cuda.is_available() :
        model.load_state_dict(torch.load(modelname ))
    else:
        model.load_state_dict(torch.load(modelname  , map_location='cpu'))
    CNNtrainer = trainer(model = model , test_dataloader=test_loader) 
    predicts = CNNtrainer.test(ensemble= True)
    return predicts
def writepredict(predict , output):
    with open(output , 'w') as f:
        subwriter = csv.writer(f , delimiter = ',')
        subwriter.writerow(["id" , "label"])
        for i in range(len(predict)):
            subwriter.writerow([str(i) , predict[i]])

if __name__ == "__main__":
    main()
