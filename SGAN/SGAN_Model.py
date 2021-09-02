# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 15:40:04 2021

@author: user
"""
import SGAN_Data as dataset

from keras.models import Model,Sequential
from keras.layers import (Conv2DTranspose,Dense,
                          Reshape,BatchNormalization,Activation,
                          Conv2D,Dropout,Flatten)
from keras.layers import LeakyReLU
from keras.utils import to_categorical
import numpy as np

class Net_models():
    def __init__(self):
        self.supervised_losses = []
        self.iterations_checkpoints = []
        self.num_classes = 10
        self.z_dim = 100
        self.img_rows = self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.discriminator_net = self.build_discriminator()
        self.discriminator_supervised = self.build_discriminator_supervised()
        self.discriminator_supervised.compile(optimizer = 'Adam',
                                         loss = 'categorical_crossentropy',
                                         metrics = ['accuracy'])
        self.discriminator_unsupervised = self.build_discriminator_unsupervised()
        self.discriminator_unsupervised.compile(optimizer = 'Adam',
                                           loss = 'binary_crossentropy')
        self.discriminator_unsupervised.trainable = False
        self.generator = self.build_generator()
        self.gan = self.build_gan()
        self.gan.compile(optimizer = 'Adam',
                loss = 'binary_crossentropy')
    def build_generator(self):
        model = Sequential()
        model.add(Dense(7*7*256,input_dim = self.z_dim))
        model.add(Reshape((7,7,256)))
        model.add(Conv2DTranspose(128,kernel_size=3,strides=2,
                                  padding='same'))#14*14*128
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(64,kernel_size=3,strides=1,
                                  padding='same'))#14*14*64
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(1,kernel_size=3,strides=2,
                                  padding='same'))#28*28*1
        model.add(Activation('tanh'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32,kernel_size=3,strides=2,
                         input_shape=self.img_shape,
                         padding='same'))#14*14*32
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(64,kernel_size=3,strides=2,
                         input_shape=self.img_shape,
                         padding='same'))#7*7*64
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(128,kernel_size=3,strides=2,
                         input_shape=self.img_shape,
                         padding='same'))#4*4*128
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes))
        return model
    def build_discriminator_supervised(self):
        model = Sequential()
        model.add(self.discriminator_net)
        model.add(Activation('softmax'))#輸出所有機率
        return model
    def build_discriminator_unsupervised(self):
        model = Sequential()
        model.add(self.discriminator_net)
        model.add(Dense(1,activation = 'sigmoid'))
        return model
    def build_gan(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator_unsupervised)
        return model

    def train(self,iterations,batch_size,sample_interval):
        real = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        num_labeled = 1000
        data = dataset.Dataset(num_labeled)
        for iterations in range(iterations):
            imgs,labels = data.batch_labeles(batch_size)
            labels = to_categorical(labels,num_classes=self.num_classes)
            imgs_unlabels = data.batch_unlabeled(batch_size)
            z = np.random.normal(0,1,(batch_size,self.z_dim))
            gen_imgs = self.generator.predict(z)
            (d_loss_supervised,accuracy) = self.discriminator_supervised.train_on_batch(imgs,labels)
            d_loss_real = self.discriminator_unsupervised.train_on_batch(imgs_unlabels,real)
            d_loss_fake = self.discriminator_unsupervised.train_on_batch(gen_imgs,fake)
            d_loss_unsupervised = 0.5 * np.add(d_loss_real,d_loss_fake)
            g_loss = self.gan.train_on_batch(z,real)
            
            if (iterations + 1) %sample_interval ==0:
                self.supervised_losses.append(d_loss_supervised)
                self.iterations_checkpoints.append(iterations + 1)
                
                print(
                    '%d [D loss supervised:%.4f,acc:%.2f%%][D loss unsupervised:%.4f [G loss :%f]'
                    %(iterations + 1,d_loss_supervised,100*accuracy,
                      d_loss_unsupervised,g_loss))
        self.model_evaluation()
    def model_evaluation(self):
        data = dataset.Dataset(100)
        x,y = data.test_set()
        y = to_categorical(y,num_classes = self.num_classes)
        _,accuracy = self.discriminator_supervised.evaluate(x,y)
        print("Test Accuracy: %.2f%%" %(100*accuracy))
            
if __name__ == '__main__':
    models = Net_models()
    iterations = 1000
    batch_size = 32
    sample_interval = 50
    models.train(iterations, batch_size, sample_interval)
    