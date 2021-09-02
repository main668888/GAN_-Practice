# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:21:59 2020

@author: user
"""
from keras.models import Model
from keras.layers import Input,Dense,Reshape,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers import Conv2DTranspose, LeakyReLU,UpSampling2D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers.merge import Add
from keras.utils.vis_utils import plot_model
import numpy as np
import cv2
import math

from keras.utils.generic_utils import Progbar

def combine_image(generated_images):
    num = generated_images.shape[0]
    #print("generated_images.shape",generated_images.shape,"num",num)
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    #print("shape",shape)
    image = np.zeros((height*shape[0],width*shape[1],shape[2]),
                    dtype = generated_images.dtype)
    
    for index,img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0],
              j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
        
    return image
'''
def generator_model():
    print("generator function")
    inputs = Input((28,28,1))
    x = Conv2D(filters = 64,kernel_size = (7,7),
               padding = 'same')(inputs)#n_height = n_width = (W-F+1)/S
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters = 64*mult*2,padding = 'same',strides = 2,
                     kernel_size = (3,3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    mult = n_downsampling*2
    for i in range(9):
        x = res_block(x,filters = 64*mult,use_dropout = True)
    
    for i in range(n_downsampling):
        #mult = 2**(n_downsampling-i)
        # x = Conv2DTranspose(filters = int(filters*mult/2),padding = 'valid',
        #                     strides = 2,
        #                     kernel_size = (3,3))(x)
        x = UpSampling2D()(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
    x = Conv2D(filters = 1,padding = 'same',
               kernel_size = (7,7))(x)
    x = Activation('tanh')(x)
    
    outputs = Add()([inputs,x])
    
    
    outputs = Lambda(lambda z: z/2)(outputs)
    
    
    model = Model(inputs=[inputs],outputs=[outputs],name = 'Generator')
    
    model.summary()
    
    plot_model(model,to_file='name.png',show_shapes=True)
    
    return model
'''
def generator_model():
    inputs = Input((10,))
    fc1 = Dense(input_dim=10, units=128*7*7)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((7, 7, 128), input_shape=(128*7*7,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    outputs = Activation('tanh')(conv2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def discriminator_model():
    print("discrimuantor function")
    inputs = Input((28,28,1))
    conv1 = Conv2D(64,(5,5),padding = 'same')(inputs) 
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1) 
    conv2 = Conv2D(128,(5,5),padding= 'same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Dropout(0.25)(conv2)#加入的
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(256,(5,5),padding= 'same')(pool2)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = Dropout(0.25)(conv3)#加入的
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    fc1 = Flatten()(pool3)
    fc1 = Dense(1)(fc1)
    fc1 = Dropout(0.5)(fc1)#加入的
    outputs = Activation('sigmoid')(fc1)
    model = Model(inputs=[inputs],outputs=[outputs])
    model.summary()
    plot_model(model,to_file='name1.png',show_shapes=True)
    
    return model
def generator_containing_discrimunator(g,d):
    d.trainable = False#凍結 不變權重
    gan_inputs = Input((28,28,1))
    x = g(gan_inputs)
    ganOutput = d(x)
    gan_Model = Model(inputs=gan_inputs,outputs=ganOutput)
    
    return gan_Model

def load_model():
    print('load_model function')
    g = generator_model() 
    d = discriminator_model()
    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss = 'binary_crossentropy', optimizer=g_optim)
    d.compile(loss = 'binary_crossentropy', optimizer=d_optim)
    d.load_weights('./weights/discriminator.h5')
    g.load_weights('./weights/generator.h5')
    return g, d

def train(Batch_size,X_train):
    print(Batch_size)
    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discrimunator(g, d)
    d_optim = RMSprop(lr=0.0006)
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss = sum_of_residual, optimizer=g_optim)# 更改損失 原本mse
    d_on_g.compile(loss = sum_of_residual, optimizer=g_optim)# 更改損失 原本mse
    d.trainable = True
    d.compile(loss=sum_of_residual, optimizer=d_optim)# 更改損失 原本mse
    
    
    for epoch in range(5):
        print("Epoch is ",epoch)
        n_iter = int(X_train.shape[0]/Batch_size)#1805/Batch_size(32)
        progress_bar = Progbar(target=n_iter)#進度條
        
        print(X_train.shape)
        for index in range(n_iter):#1805/32 = 56
            #noise
            #noise = np.random.uniform(0,1,size = (Batch_size,10))
            noise_factor = 0.5
            
            X_train_noise = X_train + noise_factor * np.random.normal(loc=0.0,
                                                                      scale=1.0, size=X_train.shape)
            
            X_train_noise_batch = X_train_noise[index*Batch_size:(index+1)*Batch_size]
            
            #load real data & fake data
            
            image_batch = X_train[index*Batch_size:(index+1)*Batch_size]#(1*32:2*32)
            #X_train_noise_batch = X_train[index*Batch_size:(index+1)*Batch_size]
            
            generated_images = g.predict(X_train_noise_batch,verbose=0)
            
            if index%20==0:
                image = combine_image(generated_images)
                image = image*127.5+127.5
                cv2.imwrite('./chest_xray/imagesnoise/'+str(epoch)+"_"+str(index)+".png",
                            image)
                
            X = np.concatenate((image_batch, generated_images))#(img , fake_img)
            y = np.array([1]*Batch_size+[0]*Batch_size)
            
            # training discriminator
            d_loss = d.train_on_batch(X,y)#先D
            d.trainable = False
            
            # training generator
            g_loss = d_on_g.train_on_batch(X_train_noise_batch, np.array([1] * Batch_size))#後G
            
            d.trainable = True 
                
            progress_bar.update(index, values=[('g',g_loss), ('d',d_loss)])
            
        print('')
        
        g.save_weights('weights/generator.h5', True)
        d.save_weights('weights/discriminator.h5', True)
        
    return d,g

def generate(Batch_size):
    g = generator_model()
    g.load_weights('./weights/generator.h5')
    noise = np.random.uniform(0,1,(Batch_size,10))
    generated_images = g.predict(noise)
    return generated_images
#loss function
def sum_of_residual(y_true,y_pred):
    #return K.mean(y_true * y_pred)
    return K.sum(K.abs(y_true-y_pred))


def feature_extractor(d=None):
    if d is None:
        d = discriminator_model()
        d.load_weights('weights/discriminator.h5')
        
    intermidiate_model = Model(inputs=d.layers[0].input,outputs=d.layers[-7].output)
    
    intermidiate_model.compile(loss = 'binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model   

### anomaly detection model define
def anomaly_detector(g=None,d=None):
    if g is None:
        g = generator_model()
        g.load_weights('weights/generator.h5')
    intermidiate_model = feature_extractor(d)
    intermidiate_model.trainable = False
    g = Model(inputs=g.layers[1].input,outputs=g.layers[-1].output)
    g.trainable = False
    
    #Input layer cann't be trained. Add new layer as same size & same distribution
    
    aInput = Input(shape=(10,))
    gInput = Dense((10),trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)
    
    # G & D feature
    G_out = g(gInput)
    D_out = intermidiate_model(G_out)
    model = Model(inputs = aInput,outputs=[G_out,D_out])
    model.compile(loss=sum_of_residual,loss_weights=[0.90, 0.10],
                  optimizer='rmsprop')
    
    K.set_learning_phase(0)

    return model

def compute_anomaly_score(model,x,interation=500,d=None):
    z = np.random.uniform(0,1,size=(1,10))
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)
    
    loss = model.fit(z,[x,d_x],batch_size = 1,epochs = interation,verbose=0)
    similar_data,_ = model.predict(z)
    
    loss = loss.history['loss'][-1]
    
    return loss,similar_data
def res_block(input_data,filters,strides=(1,1),kernel_size=(3,3),
                  use_dropout=False):
        x = Conv2D(filters = filters,strides = strides,kernel_size = kernel_size
                   ,padding = 'same')(input_data)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters = filters,kernel_size = (3,3),strides = strides,
                   padding = 'same')(x)
        x = BatchNormalization()(x)
        merged = Add()([input_data,x])
        return merged
if __name__ == '__main__':
    generator_model()
    discriminator_model()

