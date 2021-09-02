# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:11:06 2020

@author: user
"""
import numpy as np
import argparse
import AnoGANmodel
parser = argparse.ArgumentParser()
data  = np.load('traindata2.npz')
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

scr_images , tar_images = data['arr_0'],data['arr_1']

X_train = (scr_images.astype(np.float32)-127.5)/127.5
X_test = (tar_images.astype(np.float32)-127.5)/127.5
x = X_train[0]
X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]

if args.mode == 'train':
    print("Training")
    AnoGANmodel.train(32, X_train)
    
# img = AnoGANmodel.generate(36)
# y = img[1]