# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:14:56 2021

@author: user
"""
from keras.models import Model,Sequential
from keras.layers import (Input,Conv2D,Dropout,Flatten,Dense,
                          Conv2DTranspose,BatchNormalization,
                          Concatenate)
from keras.layers import (LeakyReLU,Activation)
from keras.utils import to_categorical
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import images_data as dataset
import numpy as np
import matplotlib.pyplot as plt

def createLayers(input , outputSize , kernel_size=(4,4), strides=2 ,leaky=True , batch=True , padding="same"):
    init = RandomNormal(stddev=0.02)
    l = Conv2D(outputSize ,  kernel_size = kernel_size ,  strides=strides, padding=padding , kernel_initializer=init)(input)
    if leaky:
        l = LeakyReLU(.2)(l)
    if batch:
        l = BatchNormalization()(l)
    return l
def createOutGenLayer(inLayer , outputSize , activation='relu' , norm=True , kernel_size=(4,4) , strides=(2,2)):
    init = RandomNormal(stddev=0.02)
    l = Conv2DTranspose(outputSize , activation=activation , kernel_size=kernel_size , strides=strides , padding="same" , kernel_initializer=init)(inLayer)
    if norm:
        l = BatchNormalization()(l)
    return l

class PixelGAN():
    def __init__(self,input_shape,filters=64,epochs=25,checkpoint=0):
        self._epochs = epochs + checkpoint
        self._num_filters = filters
        self.input_shape = input_shape
        self.G_model = self.build_generator()
        self.D_model = self.build_discriminator()
        self.D_model_A = self.build_discriminator_A()
        self.D_model.trainable = False
        self.D_model_A.trainable = False
        self.gan = self.build_gan()
        self.num_labeled = 900
        self.iterations = 1000
        self.num_classes = 6
        
    def build_generator(self):
        in_layer = Input(shape=self.input_shape , name = "Input")
        L1 = createLayers(in_layer , self._num_filters , batch=False)
        L2 = createLayers(L1 , self._num_filters * 2)
        L3 = createLayers(L2 , self._num_filters * 4)
        L4 = createLayers(L3 , self._num_filters * 8)
        L5 = createLayers(L4 , 100 , kernel_size=4 , strides=4)
        G5 = createOutGenLayer(L5 , self._num_filters * 4 , strides=4)
        G6 = createOutGenLayer(G5 , self._num_filters * 2)
        G7 = createOutGenLayer(G6 , self._num_filters)
        G8 = createOutGenLayer(G7 , self._num_filters)
        G9 = createOutGenLayer(G8 , 3 , strides=2  ,activation='tanh' , norm=False)
        G_model = Model(in_layer,G9,name = 'Generator_1')
        return G_model
    def build_discriminator(self):
        in_layer = Input(shape=self.input_shape , name="Discrm_Input")
        L1 = createLayers(in_layer , self._num_filters , batch=False)
        L2 = createLayers(L1 , self._num_filters * 2)
        L3 = createLayers(L2 , self._num_filters * 4)
        L4 = createLayers(L3 , self._num_filters * 8)
        
        # L5 = createLayers(L4 , 1 , kernel_size=4, strides=4 ,leaky=False , batch=False)
        L5 = Dropout(0.5)(L4)
        L6 = Flatten()(L5)
        L7 = Dense(1)(L6)
        L8 = Activation('sigmoid')(L7)
        D_model = Model(in_layer,L8,name = 'Discriminator_1')
        D_model.compile(optimizer = Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                        loss='binary_crossentropy')
        return D_model
    def build_discriminator_A(self):
        image1 = Input(shape=self.input_shape,name = 'Image_1')
        image2 = Input(shape=self.input_shape,name = 'Image_2')
        InCat = Concatenate()([image1,image2])
        L1 = createLayers(InCat , self._num_filters , batch=False)
        L2 = createLayers(L1 , self._num_filters * 2)
        L3 = createLayers(L2 , self._num_filters * 4)
        L4 = createLayers(L3 , self._num_filters * 8)
        L5 = Dropout(0.5)(L4)
        L6 = Flatten()(L5)
        L7 = Dense(1)(L6)
        L8 = Activation('sigmoid')(L7)
        D_model_A = Model([image1,image2],L8)
        D_model_A.compile(optimizer = Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                        loss='binary_crossentropy')
        return D_model_A
        
    def build_gan(self):
        imgs = Input(shape = self.input_shape)
        re_images = self.G_model(imgs)
        validity = self.D_model(re_images)
        validity_A = self.D_model_A([imgs,re_images])
        model = Model(imgs,[re_images,validity,validity_A])
        model.compile(optimizer = Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                      loss=['mse','binary_crossentropy','binary_crossentropy'],
                      loss_weights=[0.99, 0.005,0.005])
        return model
    def sample_imagess(self,epoch,x_test,generator_mnist):
        r, c = 5, 5
        gen_imgs = generator_mnist.predict(x_test)
        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        # cnt  = np.random.randint(0,x_test.shape[0]-1,size=1)
        cnt = 0
        print('x_test_label:',cnt)
        
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("image_T/T_%d.png" % epoch)
        plt.close()
    
    
    def train(self,batch_size,sample_interval):
        real = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        data = dataset.Dataset(self.num_labeled)
        
        for iterations in range(self.iterations):
            imgs,labels,imgs_T = data.batch_labeles(batch_size)
            real_images = imgs
            labels = to_categorical(labels,num_classes=self.num_classes)
            fake_images = self.G_model.predict(real_images)
            d_loss_real = self.D_model.train_on_batch(real_images, real)
            d_loss_real_T = self.D_model.train_on_batch(imgs_T, real)
            d_loss_fake = self.D_model.train_on_batch(fake_images, fake)
            
            d_loss_A_fake = self.D_model_A.train_on_batch([imgs,fake_images],fake)
            d_loss_A_dis = self.D_model_A.train_on_batch([imgs,imgs],fake)
            d_loss_A_T = self.D_model_A.train_on_batch([imgs,imgs_T],real)
            
            d_loss = np.add(d_loss_real_T,np.add(d_loss_real, d_loss_fake))/3
            
            d_loss_A = np.add(d_loss_A_T,np.add(d_loss_A_dis, d_loss_A_fake))/3
            
            g_loss = self.gan.train_on_batch(real_images,[imgs_T,real,real])
            if (iterations + 1) %sample_interval ==0:
                print(
                    '%d [D loss unsupervised:%.4f] [G loss :%f][D_loss_A : %f]'
                    %(iterations + 1,d_loss,g_loss[0],d_loss_A))
                self.sample_imagess(iterations,real_images,self.G_model)
                
        imgs,_,_ = data.batch_labeles(batch_size)
        print('iterations:',iterations)
        self.sample_imagess(iterations+1,real_images,self.G_model)
        gen_images = self.G_model.predict(imgs)
        plt.figure(1, figsize=(20, 20))
        plt.subplot(2,2,1)
        plt.title('gan_images')
        plt.imshow(gen_images[10].reshape(64,64,3))
        plt.subplot(2,2,2)
        plt.title('bag_images')
        plt.imshow(imgs[10].reshape(64,64,3))
        plt.subplot(2,2,3)
        plt.title('gan_images')
        plt.imshow(gen_images[20].reshape(64,64,3))
        plt.subplot(2,2,4)
        plt.title('bag_images')
        plt.imshow(imgs[20].reshape(64,64,3))
        plt.show()
        
x = PixelGAN(input_shape=(64,64,3))
x.D_model.summary()
x.train(batch_size = 32, sample_interval = 50)



