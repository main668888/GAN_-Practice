# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:24:18 2021

@author: user
"""
from keras.datasets import mnist
import numpy as np
class Dataset():
    def __init__(self,num_labeled):
        self.num_labeled = num_labeled
        (self.x_train,self.y_train),(self.x_test,self.y_test) = mnist.load_data()
        
        def preprocess_imgs(x):
            x = (x.astype(np.float32) - 127.5) / 127.5
            x = np.expand_dims(x, axis=3)
            return x
        def preprocess_labels(y):
            return y.reshape(-1,1)
        
        self.x_train = preprocess_imgs(self.x_train)
        self.y_train = preprocess_labels(self.y_train)
        
        self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(self.y_test)
    def batch_labeles(self,batch_size):
        idx = np.random.randint(0,self.num_labeled,batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs,labels
    def batch_unlabeled(self, batch_size):
        idx = np.random.randint(self.num_labeled,self.x_train.shape[0],batch_size)
        imgs = self.x_train[idx]
        return imgs
    
    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train,y_train
    def test_set(self):
        return self.x_test,self.y_test


if __name__ == '__main__':
    num_labeled = 100
    x = Dataset(num_labeled)
    imgs,labels = x.batch_labeles(32)