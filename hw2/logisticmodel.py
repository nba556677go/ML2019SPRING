import csv
import numpy as np
import sys

class logisticmodel():
    def __init__(self ):
        pass
        #self.normalize = normalize
    def initpara(self , train_x , l_rate):
        self.l_rate = l_rate
        self._w = np.zeros(train_x.shape[1])
        self._sgrad = np.zeros(train_x.shape[1])
    def add_bias(self , train_x):
        return np.concatenate( (np.ones((train_x.shape[0] , 1)) , train_x), axis = 1)
    def train(self ,train_x , train_y, l_rate = 0.1, epochs = 100000, batch = None , validation_percent = 0, save = 5000 , regularize = 0.0):
        self._lambdda = regularize
        
        if validation_percent !=0:
            train_x, train_y , val_x, val_y = split_valid(train_x, train_y , validation_percent)        
        #add bias
        train_x = self.add_bias(train_x)
        
        #init w, sgrad
        self.initpara(train_x, l_rate)

        


        for epoch in range(1, epochs + 1):
            self.step(train_x , train_y)

            if (epoch % 500 == 0):
                acc = self.predict(train_x , train_y)
                print('[Epoch {:5d}] - training loss: {:.5f}, accuracy: {:.5f}'.format(epoch, self._loss, acc))
            if (epoch % save == 0):
                np.save("logistic-feature6-"+ str(epoch)+ ".npy", self._w)
            if validation_percent is not None:
                  #  print('\tvalid loss: {:.5f}, accuracy: {:.5f}'.format(self._loss(X_valid, Y_valid), self.evaluate(X_valid, Y_valid)))
                pass

        np.save("logistic-feature6"+str(epochs)+"-regularize-"+str(self._lambdda)+".npy" , self._w)

    def step(self , x , y):
         #self.update(x , y , self.sigmoid(np.dot(x , self._w)))
        func = self.sigmoid(np.dot(x , self._w))
        self.update(x , y , func)
    
    def update(self ,x , y , func):
        grad = -np.dot(x.transpose(), (y - func))  + self._lambdda*np.sum(self._w)
        #print("hypo :" , hypo.shape)
        
        self._sgrad += grad**2
        ada = np.sqrt(self._sgrad)
        self._w = (1- self.l_rate*self._lambdda)*self._w - self.l_rate * grad/ada
    

       
    def cross_entropy(self , x , y , func):
        regular = np.sum(self._w**2)
        return -np.mean(y * np.log(func + 1e-20) + (1 - y) * np.log(1 - func + 1e-20))+ np.mean(self._lambdda*0.5*regular)


    def predict(self , x , y):
        
        self._loss = self.cross_entropy(x , y , self.sigmoid(np.dot(x , self._w)))
        pred = self.sigmoid(np.dot(x , self._w))
        p = pred
        p[pred < 0.5] = 0.0
        p[pred >= 0.5] = 1.0
        return np.mean(1 - np.abs(y - p))

    



    @staticmethod
    def sigmoid(z):
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, 1e-8, 1-(1e-8))
    @staticmethod
    def split_valid(train_x, train_y, validation_percent):
        pass
    