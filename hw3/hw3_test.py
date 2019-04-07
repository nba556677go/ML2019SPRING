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
from model import CNN
import torchvision.transforms as transforms

####################
batch_size = 128
num_emsemble = 5
def main():
    test_transform = transforms.Compose([
                transforms.ToPILImage()  ,
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5,), std = (0.5, )),
        ])
    test_dataset = MyDataset(testx_file= sys.argv[1] , 
                                is_train = False , save = False , transform= test_transform)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False) 
    predict_list = []
    for i in range(num_emsemble):
        predict_list.append(ensemble(modelname = str(i+1)+'.pt', test_loader = test_loader))
   # print(predict_list)
    predict_list = np.array(predict_list)
    print(predict_list.shape)
    #input()
    #mean
    predict_list = predict_list.mean(axis = 0)
   # predict_list = torch.Tensor(predict_list)
   # output = torch.sum(predict_list , dim = 0)
    predicts = predict_list.argmax(axis = 1)
   # predicts = torch.max(output, dim =1)[1] 
    writepredict(predict = predicts , output = sys.argv[2])
    

def ensemble(modelname,test_loader ):
    model = CNN()
    model.load_state_dict(torch.load(modelname ))
    CNNtrainer = trainer(model = model , test_dataloader=test_loader) 
    predicts = CNNtrainer.test(ensemble= True)
    #model.eval()
    return predicts
    
  
def writepredict(predict , output):
    #flat_list = [item for sublist in predict for item in sublist]
    with open(output , 'w') as f:
        subwriter = csv.writer(f , delimiter = ',')
        subwriter.writerow(["id" , "label"])
        for i in range(len(predict)):
            subwriter.writerow([str(i) , predict[i]])

if __name__ == "__main__":
    main()
