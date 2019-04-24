
import csv 
import numpy as np
import math
import sys
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torchvision.models as models
import os
#import scipy.misc
from torch_data import hw5Dataset
from  PIL import Image
#############hyper parameters#########
__CUDA__ = torch.cuda.is_available()
data_dir = sys.argv[1]
base_dir = './'
outdir = os.path.join(base_dir, sys.argv[2])
if not os.path.exists(outdir):
  os.mkdir(outdir)
eps = 5e-4
steps =70
norm = float('inf')
step_alpha = 0.01
batchsize = 8
##################################################
def get_Linf(outname ,data_dir):
    out = Image.open(outname )
    out = np.array(out, dtype='float32').flatten()
    #out = out.flatten()
   # print(out)
    datas = os.listdir(data_dir)
    print(outname[len(outdir)+len(base_dir)-1:])
    for i in datas:
        if i == outname[len(outdir)+len(base_dir)-1:] or i ==outname[len(outdir)+len(base_dir):]:
            max_diff = 0
            origin = Image.open(os.path.join(data_dir ,i))
            origin = np.array(origin , dtype='float32').flatten()
            result = abs(out-origin)
            print("maxdiff:" ,np.max(result))
    return 
def main():
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage(),
        ])

    #load data
    dataset = hw5Dataset(
        save=True,
        #loadfiles= 'images.pkl',
        train_file=data_dir,
        label_file="train_y.npy",
        transform = data_transform
    )

    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
   
    #loss_fn = nn.CrossEntropyLoss()
    model = models.resnet50(pretrained=True)


    if __CUDA__:
        model.cuda()
        loss_fn =  nn.CrossEntropyLoss().cuda()
    else:
        model.cpu()
        loss_fn =  nn.CrossEntropyLoss().cpu()

    model.eval()
    ########################################
    #attacking process
    train_acc = []
    index = '0'
    for i, (x, y) in enumerate(dataloader):          # for each training step
        print('Batch:',i+1)
        #print(x)
        #print(y)

        if __CUDA__:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x, requires_grad=True), Variable(y)
        #result = 0
        for step in range(steps):
            output = model(x)
            #print("output" ,output)
            # print(x)
            model.zero_grad()
            loss = loss_fn(output,y)
            loss.backward()
            grad = x.grad.data

            normed_grad = step_alpha * torch.sign(grad)
            step_adv = x.data + normed_grad
            adv = step_adv - x
            adv = torch.clamp(adv, -eps, eps)
            result = x + adv
            #print(result.size)
            for i in range(len(result)):
                result[i][0] = torch.clamp(result[i][0], -0.485/0.229, 0.515/0.229)
                result[i][1] = torch.clamp(result[i][1], -0.456/0.224, 0.544/0.224)
                result[i][2] = torch.clamp(result[i][2], -0.406/0.225, 0.594/0.225)
            
            x.data = result
        #print(result.size())
        #print(result)
        #print(result.detach().cpu().view(3,224,224))
        #input()
        for i in range(len(result)):
            output_image = inverse_transform(result[i].detach().cpu().view(3,224,224))
        
        #output_image *= 225
            #print(np.array(output_image))
            #print(np.array(output_image).shape)
            #input()
        #get_Linf(output_image, i)
        #index = str(i)
            index = str(index)
            while len(index) < 3:
                index = '0' + index
            filename = os.path.join(outdir , index + '.png')
            output_image.save(filename)
        #scipy.misc.imsave(filename, np.array(output_image))
            get_Linf(filename ,data_dir)
            index = int(index)
            index += 1
        pre = model(result)
        predict = torch.max(pre, 1)[1]
        #print('predicted:{}, ground_truth:{}'.format(predict.item(), y.item()))
        acc = np.mean((y == predict).cpu().numpy())
        #print('Train_acc: {:.6f}, Success_rate: {:.6f}'.format(np.mean(train_acc), 1.0-np.mean(train_acc)))
        train_acc.append(acc)
        print('Success_rate: {:.6f}'.format(1.0-np.mean(train_acc)))
        
    print('Train_acc: {:.6f}, Success_rate: {:.6f}'.format(np.mean(train_acc), 1.0-np.mean(train_acc)))
    
    #get_Linf(outdir ,data_dir)
if __name__ == "__main__":
    main()





    
