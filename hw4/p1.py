import csv 
import numpy as np
import math
import sys
from torch_data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
from trainer_hw4 import trainer
from model import CNN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from cutdata import loadval
import os
####################
batch_size = 128
num_emsemble = 1
validation_len = 2870
class_index = [0,299,2,7,3,15,4]
def main():

    """
    test_transform = transforms.Compose([
                transforms.ToPILImage()  ,
                tranforms.ToTensor(),
                transforms.Normalize(mean = (0.5,), std = (0.5, )),
        ])
    """
    """
    val_x = np.load("../hw3/train_x.npy")
    val_x = val_x[:validation_len,:,:,:]
    print(val_x.shape)
    np.save("val_x.npy" , val_x)
    val_y = np.load("../hw3/train_y.npy")
    val_y = val_y[:validation_len]
    print(val_y.shape)
    np.save("val_y.npy" , val_y)
    sys.exit()
    """
    val_x , val_y = loadval(sys.argv[1])
    #val_x = np.load("val_x.npy")
    #val_y = np.load("val_y.npy")
    val_x = val_x.reshape(val_x.shape[0] , 1 , 48 , 48)
    val_x = val_x[class_index ,: ,: ,:]
    #print(val_x)
    #print(val_x.shape)
    val_x = torch.Tensor(val_x)
    val_y = val_y[class_index]
    #print(val_y)
    val_y = torch.LongTensor(val_y)
    
    model = CNN()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("notrans.pt"))
    else:
        model.load_state_dict(torch.load("notrans.pt" , map_location='cpu')   )
    #p1
    show_saliency_maps(val_x, val_y, model)
    

def compute_saliency_maps(x, y, model):
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()
    x.requires_grad_()
    loss_func = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        y_pred = model(x.cuda())    
        loss = loss_func(y_pred, y.cuda())
    else:
        y_pred = model(x.cpu())    
        loss = loss_func(y_pred, y.cpu())        
    loss.backward()
    
    predict = torch.max(y_pred, 1)[1]
    print(predict)
    saliency = x.grad.abs().squeeze().data
    return saliency
     

    
def show_saliency_maps(x, y, model):
    print(x.shape)
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(x, y, model)
    x_org = x.squeeze().detach().numpy()
    print(x_org.shape)
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    
    num_pics = x_org.shape[0]
    base_dir = './'
    outdir = os.path.join(base_dir, sys.argv[2])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in range(num_pics):
        # You need to save as the correct fig names
        plt.figure()
        plt.imshow( x_org[i], cmap=plt.cm.gray)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()      
        fig.savefig(os.path.join(outdir , 'fig1_'+ str(i)), dpi=100)

        plt.figure()
        plt.imshow(saliency[i], cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(outdir , 'saliency_'+ str(i)), dpi=100)

        if i == 3:
            thres = 0.000000025
        elif i == 5:
            thres = 0.0001
        elif i == 4:
            thres = 0.0005
        else :
            thres = 0.001
        see = x_org[i]
        see[np.where(saliency[i] <= thres)] = np.mean(see)
        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(outdir , 'see_'+ str(i)), dpi=100)




if __name__ == "__main__":
    main()
