# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:04:37 2021

@author: user
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:38:53 2021

@author: user
"""
from Generator_data import trainGenerator,test_image
from keras.layers.merge import _Merge
from keras.utils.vis_utils import plot_model
from keras.models import  Model,Sequential
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D,UpSampling2D,AveragePooling2D
from keras.layers import Dense,Dropout,Reshape
from keras.layers import BatchNormalization,Activation,Flatten
from keras.layers import Lambda,concatenate
from keras.layers import Layer
import keras.backend as K
from keras.optimizers import Adam,RMSprop
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
from functools import partial
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve, auc
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
class CntLoss(Layer):
    def __init__(self,**kwargs):
        super(CntLoss,self).__init__(**kwargs)
    def call(self,x,mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],1)
class LatLoss(Layer):
    def __init__(self, **kwargs):
        super(LatLoss,self).__init__(**kwargs)
    def call(self,x,mask=None):
        ori = feature(x[0])
        gan = feature(x[1])
        return K.mean(K.square(ori - gan))
    def get_output_shape_for(self,input_shape):
        return (input_shape[0][0],1)
class Net_models():
    def __init__(self):
        self.img_shape = (128, 128, 1)
        self.n_critic = 5
        self.clip_value = 0.01
        self.data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
        # Build and compile the discriminator
        self.optimizer = RMSprop(lr=0.001)
        self.generator = self.build_generator()
        self.feature = self.build_featrue()
        self.discriminator = self.build_discriminator(self.feature)
        self.discriminator.compile(optimizer = 'adam',loss = self.wasserstein_loss)
        self.discriminator.trainable = False
    def build_model(self):
        
        real_img = Input(shape =self.img_shape)
        vaild = self.discriminator(real_img)
        gan_image = self.generator(real_img)
        cnt_loss = CntLoss(name='cnt_loss')([real_img,gan_image])
        lat_loss = LatLoss(name='lat_loss')([real_img,gan_image])
        gan_trainer = Model(real_img, [cnt_loss,lat_loss,vaild])
        losses = {'cnt_loss':self.loss,'lat_loss':self.loss,'discriminator':self.wasserstein_loss}
        lossWeights = {'cnt_loss':20,'lat_loss':1,'discriminator':1}
        gan_trainer.compile(optimizer = 'adam',loss = losses,
                            loss_weights = lossWeights)
        return gan_trainer
    def loss(self,yt,yp):
        return yp
    def wasserstein_loss(self,y_label, y_pred):
        return K.mean(y_label * y_pred)
    def encoder_layer(self,inputs,filters,BN,
                      kernel_size=3,Drop=False,activation='relu'):
        x = inputs
        x = Conv2D(filters,kernel_size,padding='same',
                   kernel_initializer = 'he_normal',
                   bias_initializer = 'zero')(x)
        x = Conv2D(filters,kernel_size,padding='same',
                   kernel_initializer = 'he_normal',
                   bias_initializer = 'zero')(x)
        if BN==True:
            x = BatchNormalization(momentum=0.8)(x)
        if activation=='leakyReLU':
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = Activation('relu')(x)
        if Drop == True:
            x = Dropout(0.3)(x)
        return x
    def decoder_layer(self,inputs,paired_inputs,filters,connect,kernel_size=3):
        x = inputs
        x = Conv2D(filters, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        if connect==True:
             x = concatenate([paired_inputs,x])
        x = Conv2D(filters,kernel_size,padding='same'
                        ,kernel_initializer = 'he_normal',
                        bias_initializer = 'zero')(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = Activation('relu')(x)
        return x
    def build_generator(self):
        img = Input(shape = self.img_shape,name = 'input_layer')
        E1 = self.encoder_layer(img,32,BN=False,activation='leakyReLU')
        P_1 =AveragePooling2D((2, 2))(E1)
        E2 = self.encoder_layer(P_1,64,BN=True,activation='leakyReLU')
        P_2 = AveragePooling2D((2, 2))(E2)
        E3 = self.encoder_layer(P_2,128,BN=True,activation='leakyReLU')
        P_3 = AveragePooling2D((2, 2))(E3)
        E4 = self.encoder_layer(P_3,256,BN=True,Drop=True,activation='leakyReLU')
        # P_4 = AveragePooling2D((2, 2))(E4)
        G_1 = GlobalAveragePooling2D(name='encoder_output')(E4)
        D_1 = Dense(128*128*2)(G_1)
        R_1 = Reshape((128//8, 128//8, 128))(D_1)
        D_1 = self.decoder_layer(R_1,E4,256,connect = True)
        Up_1 = UpSampling2D(size = (2,2))(D_1)
        D_2 = self.decoder_layer(Up_1,E3,128,connect = True)
        Up_2 = UpSampling2D(size = (2,2))(D_2)
        D_3 = self.decoder_layer(Up_2,E2,64,connect = True)
        Up_3 = UpSampling2D(size = (2,2))(D_3)
        D_4 = self.decoder_layer(Up_3,E1,32,connect = True)
        C_1 = Conv2D(2,3,padding='same')(D_4)#20210331改過原來 是ReLU
        C_1 = LeakyReLU(alpha=0.2)(C_1)
        C_2 = Conv2D(1,3,activation='tanh',padding='same')(C_1)
        x = Model(img,C_2,name='generator_u')
        return x
    def build_featrue(self):
        img = Input(shape = self.img_shape)
        E1 = self.encoder_layer(img,32,BN=False,activation='leakyReLU')
        P_1 = AveragePooling2D((2, 2))(E1)
        E2 = self.encoder_layer(P_1,64,BN=True,activation='leakyReLU')
        P_2 = AveragePooling2D((2, 2))(E2)
        E3 = self.encoder_layer(P_2,128,BN=True,activation='leakyReLU')
        P_3 = AveragePooling2D((2, 2))(E3)
        E4 = self.encoder_layer(P_3,256,BN=True,Drop=True,activation='leakyReLU')
        # P_4 = AveragePooling2D((2, 2))(E4)
        G_1 = GlobalAveragePooling2D(name='glb_avg_1')(E4)
        f = Dense(100,activation='relu',name='latent_z')(G_1)
        f = Model(img,f,name='feature')
        return f
    def build_discriminator(self,feature_model):
        img = Input(shape = self.img_shape)
        f = feature_model(img)
        d = Dense(1,name = 'd_out')(f)
        d = Model(img,d,name = 'discriminator')
        return d
    def train(self,batch_size,epochs,sample_interval):
        gan_trainer = self.build_model()
        valid = np.ones((batch_size,1))*(0.9)+(0.1/2)#更改過本來為+0.1
        fake = -np.ones((batch_size,1))*(0.9)+(0.1/2)
        myGene = trainGenerator()
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                x = myGene.__next__()
                fake_images = self.generator.predict(x)
                d_loss_real = self.discriminator.train_on_batch(x,valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_images,fake)
                d_loss = 0.5*np.add(d_loss_real,d_loss_fake)
                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight, -self.clip_value,
                                      self.clip_value) for weight in weights]
                    layer.set_weights(weights)
            g_loss = gan_trainer.train_on_batch(x,[valid,valid,fake])
            if epoch % sample_interval ==0:
                lr_g = K.get_value(gan_trainer.optimizer.lr)
                lr_d = K.get_value(self.discriminator.optimizer.lr)
                print('lr_g:',lr_g)
                print('lr_d:',lr_d)
                lr_g,lr_d = self.CalculationLR(epoch, lr_g,lr_d)
                K.set_value(gan_trainer.optimizer.lr, lr_g)
                K.set_value(self.discriminator.optimizer.lr, lr_d)
                lr_g_after = K.get_value(gan_trainer.optimizer.lr)
                lr_d_after = K.get_value(self.discriminator.optimizer.lr)
                print('lr_g_after:',lr_g_after)
                print('lr_d_after:',lr_d_after)
                self.sample_images(epoch,x)
                print(f'niter: {epoch+1}, g_loss: {g_loss}, d_loss: {d_loss}')
            
        test_data = test_image()
        x_test,y_test = test_data.__next__()
        
        scores,gan_images = self.evaluation(self.feature,self.generator,x_test,y_test)
        self.plot_all_image(scores,y_test,x_test,gan_images)
    def CalculationLR(self,epoch,lr_g,lr_d):
        # if epoch == 200 or epoch == 300 or epoch == 400 or epoch == 500 or epoch == 600 or epoch == 700:
        lr_g = 0.5*lr_g
        lr_d = 0.5*lr_d
        return lr_g,lr_d
    def evaluation(self,feature,generator,images,labels):
        print('image.shape:',images.shape)
        print('label.shape:',labels.shape)
        gan_images = generator.predict(images)
        real_feature = feature.predict(images)
        gan_feature = feature.predict(gan_images)
        images = images.reshape(images.shape[0],(images.shape[1]*images.shape[2]*images.shape[3]))
        gan_images = gan_images.reshape(gan_images.shape[0],(gan_images.shape[1]*gan_images.shape[2]*gan_images.shape[3]))
        
        rec_mean = np.mean(np.abs(images-gan_images),axis = 1)
        lat_mean = np.mean(np.square(real_feature-gan_feature),axis = 1)
        lat = np.square(real_feature-gan_feature)
        score = 0.1*rec_mean+0.9*lat_mean
        score = (score - np.min(score))/(np.max(score)-np.min(score))
        
        X_tsne = manifold.TSNE(n_components = 3,init='random',
                               random_state=5,verbose=1).fit_transform(lat)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne-x_min)/(x_max-x_min)
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2],
               c= ['pink'if x == 1 else 'skyblue' for x in labels])
        plt.show()
        return score,gan_images

    def plot_all_image(self,score,labels,real_images,gan_images):
        one_number = []
        zero_number = []
        plt.figure(1,figsize=(14, 5))
        plt.scatter(range(len(score)),score,c = ['pink'if x == 1 else 'skyblue' for x in labels])
        for i in range(len(score)):
            if labels[i] == 1:
                one_number = np.append(one_number,score[i])
            else:
                zero_number = np.append(zero_number,score[i])
        plt.figure(2,figsize=(7,7))
        plt.hist(one_number,label='abnormal',stacked = True,density = True,
                 alpha = 0.5,color = 'r')
        plt.hist(zero_number,label='normal',stacked = True,density = True,
                 alpha = 0.5,color = 'b')
        plt.legend()
        fpr,tpr,_ = roc_curve(labels, score,pos_label = 1)
        roc_auc = auc(fpr,tpr)
        plt.figure(3,figsize=(10, 10))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        n = 10 
        plt.figure(4,figsize=(20,4))
        for i in range(n):
            ax = plt.subplot(2,n,i+1)
            plt.imshow(real_images[i].reshape(128,128))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            ax = plt.subplot(2,n,i+1+n)
            plt.imshow(gan_images[i].reshape(128,128))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    def sample_images(self,epoch,x):
        gan_imgs = self.generator.predict(x)
        gan_imgs = 0.5 * gan_imgs + 0.5
        plt.figure(2,figsize=(10, 10))
        for i in range(16):
            ax = plt.subplot(4,4,i+1)
            plt.imshow(gan_imgs[i].reshape(128,128),cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("image_test_no_LR/A_real_images_%d.png" % epoch)
        plt.close()
if __name__ == '__main__':
    for i in range(10):
        UNET_GAN = Net_models()
        feature = UNET_GAN.feature
        gan_model = UNET_GAN.build_model()
        # UNET_GAN.generator.summary()
        # plot_model(gan_model,to_file = './Test_model/gans_model.png',show_shapes=True)
        UNET_GAN.train(batch_size = 32, epochs = 1000, sample_interval = 100)