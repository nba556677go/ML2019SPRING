import torch.nn as nn
import torch
# define our model
#VGG16 https://ithelp.ithome.com.tw/articles/10192162
class CNN(nn.Module):
    def __init__(self , num_classes = 7 ):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(  # input shape (1, 48, 48)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=64,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape = (𝑊−𝐹+2𝑃)/𝑆+1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(
                in_channels=64,      
                out_channels=64,    
                kernel_size=3,      
                stride=1,          
                padding=1,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape = (𝑊−𝐹+2𝑃)/𝑆+1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),    # activation,

            nn.MaxPool2d(kernel_size=2 , stride=2 ),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(64, 128 , 3, 1, 1),  # output shape 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # activation
            nn.Conv2d(128, 128 , 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2 , stride = 2),  # output shape 
        )
        self.conv3 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(128, 256 , 3, 1, 1),  # output shape (32, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # activation
            nn.Conv2d(256, 256 , 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256 , 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2 , stride = 2),  # output shape (32, 7, 7)
        )
        #self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        self.conv4 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(256, 512 , 3, 1, 1),  # output shape (32, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # activation
            nn.Conv2d(512, 512 , 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512 , 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2 , stride = 2),  # output shape (32, 7, 7)
        )
        self.conv5 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(512, 512 , 3, 1, 1),  # output shape (32, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512 , 3, 1, 1),  # output shape (32, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512 , 3, 1, 1),  # output shape (32, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2 , stride = 2),
        )        

        self.dense = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(512*1*1, 128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),           
            nn.Linear(32, num_classes),
        )
        #self.output = nn.Softmax(dim =1)
    
    def forward(self, x):
        # You can modify your model connection whatever you like
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
       # print(x.size())
        out = self.dense(x.view(-1,  512*1*1)    )#flatten
        #out = self.output(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]