import csv 
import numpy as np
import math
import sys
from torch_data import MyDataset , get_feature3
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import torch.nn as nn
import torch
import operator

# Hyper-parameters 
input_size = 103
num_classes = 2
num_epochs = 54
batch_size = 1024
l_rate = 0.001
validation_split = 0.1
final_model_name = 'nn-epoch'+str(num_epochs)+'.pt'
can_train = False
is_validation = 0
ensemble_num = 5
device = torch.device('cpu')
# define our model
class Model(nn.Module):
    def __init__(self , feature_size):
        super(Model, self).__init__()
        self.feature_size = feature_size
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            #torch.nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        self.output = nn.Softmax(dim =1)
    
    def forward(self, x):
        # You can modify your model connection whatever you like
        out = self.fc(x.view(-1, self.feature_size))
        out = self.output(out)
        return out

def validation(model , valid_loader):
    valid_acc = []
    for _, (data_x, target) in enumerate(valid_loader):
        data = data_x.to(device = device ,  dtype=torch.float)
        target_valid = target.to(device = device ,  dtype=torch.long)  
        output = model(data)
        predict = torch.max(output, 1)[1]
        #predict = predict.float()
        acc = np.mean((target_valid == predict).numpy())
        valid_acc.append(acc)
    return valid_acc

def writepredict(predict , output):
    #flat_list = [item for sublist in predict for item in sublist]
    with open(output , 'w') as f:
        subwriter = csv.writer(f , delimiter = ',')
        subwriter.writerow(["id" , "label"])
        for i in range(len(predict)):
            if predict[i] < 0.5 :
                subwriter.writerow([str(i+1) , 0])
            else:
                subwriter.writerow([str(i+1) , 1])
def predict( model   , model_name , test_loader    , output = None , device=device):
    #model = Model(feature_size = feature_size)
# test
    
    model.load_state_dict(torch.load(model_name))
    model.eval()
    predict_list = []
    
    #test_x = loadtest(sys.argv[3] , mean = train_dataset.mean , std = train_dataset.std)
    for _, data_x in enumerate(test_loader):
        data = data_x.to(device = device ,  dtype=torch.float)
        out = model(data)
        predict = torch.max(out, 1)[1]
        predict_w = np.array(predict.float())
        #predict_w = np.transpose(predict_w)
        #print(predict_w.shape)
        #print(predict_list.shape)
        for i in predict_w:
            predict_list.append(i)
        #count += 1
    #w = np.load(sys.argv[3])

    #print(count)
    #print(predict_list)
    if output != None: 
        writepredict(predict = predict_list , output = output)
    else:
        return predict_list
def ensemble(sol , output ):

    final=[]
    #print(w1)
    #assert(len(sol[0]) == len(sol[3]) and len(sol[4]) == len(sol[0]))
    #print(len(sol[0]))
    #input()
    for i in range(len(sol[0])):
        dic ={0:0 , 1: 0 }
        for j in range(len(sol)):
            dic[sol[j][i]] += 1

        #print(dic)
        #input()
        maxi = max(dic.items(), key=operator.itemgetter(1))[0]
        final.append(maxi)
        #print(i , maxi)

    writepredict(predict = final , output = output)
if __name__ == "__main__":
    
    #train_x , train_y = loaddata(sys.argv[1] , sys.argv[2] , 1)
    train_dataset = MyDataset(trainx_file =sys.argv[1] , trainy_file =sys.argv[2] , train = True )
    #train_loader = DataLoader(dataset=train_dataset,
    #                    batch_size=batch_size,
    #                    shuffle=True)
    test_dataset = MyDataset(trainx_file =  None ,trainy_file = None, testx_file= sys.argv[3] , 
                            train = False , mean = train_dataset.mean , std = train_dataset.std )
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False) 
    
    if is_validation == 0:
        train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True)
    else:
        #creating validation set
        dataset_len = len(train_dataset)
        indices = list(range(dataset_len))
        val_len = int(np.floor(validation_split * dataset_len))
        validation_idx = np.random.choice(indices, size=val_len, replace=False)
        train_idx = list(set(indices) - set(validation_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)
        print("len train:" , len(train_idx))
        print("len valid:" , len(validation_idx))
        #print(train_idx)
        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                             sampler = train_sampler)
        validation_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            sampler = validation_sampler)
        #data_loaders = {"train": train_loader, "val": validation_loader}
        #data_lengths = {"train": len(train_idx), "val": val_len}
        earlystop_count = 0
        tot_valid = [0.0]

    
    
    if can_train == True:

        total_predict = []
        
        
        for i in range(1 , ensemble_num+1):
            model = Model(train_dataset.feature_size)
            
            model.to(device)
            optimizer = Adam(model.parameters(), lr=l_rate , weight_decay = 0.00001 )
            loss_fn = nn.CrossEntropyLoss()
            model.train()
            for epoch in range(num_epochs):
                train_loss = []
                train_acc = []
                """
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        #optimizer = scheduler(optimizer, epoch)
                        model.train(True)  # Set model to training mode
                    else:
                        model.train(False)  # Set model to evaluate mode
                running_loss = 0.0
                """
                for _, (data_x, target) in enumerate(train_loader):
                    data = data_x.to(device = device ,  dtype=torch.float)
                    target_cpu = target.to(device = device ,  dtype=torch.long)
                    #print("1batch shape: " , data.size())
                    
                    # You can also use
                    # data = img.cuda()
                    # target_cuda = target.cuda()
                    
                    
                    optimizer.zero_grad()
                    
                    output = model(data)
                    loss = loss_fn(output, target_cpu)
                    #if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                    predict = torch.max(output, 1  )[1]
                    
                    #print(target_cpu.type())
                    #predict = predict.float()
                    #print(predict.type())
                    acc = np.mean((target_cpu == predict).numpy())
                    
                    train_acc.append(acc)
                    train_loss.append(loss.item())
                    #running_loss += loss.data[0]
                
                #validation
                if is_validation == 1:
                    model.eval()
                    valid_acc = validation(model , validation_loader ) 
                    model.train() 
                    print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, valid_Acc: {:.4f}".format(epoch, np.mean(train_loss), np.mean(train_acc) , np.mean(valid_acc)))
                    
                    #print(tot_valid)
                    if np.mean(valid_acc) > max(tot_valid):
                        torch.save(model.state_dict(), 'nn-epoch'+str(epoch+1)+'.pt')
                        tot_valid.append(np.mean(valid_acc))
                else:
                    print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch+1, np.mean(train_loss), np.mean(train_acc) ))
                
                    #torch.save(model.state_dict(), 'nn-epoch'+str(epoch+1)+'.pt')
            if is_validation == 0:        
                torch.save(model.state_dict(), 'nn-epoch'+str(num_epochs)+"ensemble-"+str(i)+'.pt')
        





    ################################testing#################################
    name = []
    tot_predict = []
    for i in range(1 ,ensemble_num+1):
        name.append('ensemble/nn-epoch'+str(num_epochs)+"ensemble-"+str(i)+'.pt')
    for i in name:
        model = Model(train_dataset.feature_size)
        tot_predict.append(predict(model = model, model_name = str(i)  , test_loader = test_loader ))
        #print(len(tot_predict[-1]))
    
    ensemble(sol = tot_predict ,output= sys.argv[4])

    