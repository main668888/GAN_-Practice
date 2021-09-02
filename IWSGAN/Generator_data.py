# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:41:09 2021

@author: user
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import os
import skimage.transform as trans
import glob
import matplotlib.pyplot as plt
def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img
def trainGenerator():
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.5,
                         horizontal_flip=True,
                         preprocessing_function = prep_fn,
                         fill_mode='nearest')
    image_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
            './Data_images/image_train_gun',
            classes = ['image'],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = (128,128),
            batch_size = 32,
            save_to_dir = './Data_images/image_train_gun/Enhanced_image', # './Data_images/image_train_gun/Enhanced_image'
            save_prefix  = 'Enhanced image',
            seed = 1)
    return image_generator

def test_image():
    image_test = ImageDataGenerator(preprocessing_function = prep_fn)
    image_test_generator = image_test.flow_from_directory(
        './Data_images/image_test_2_gun',
        classes = ['normal','xabnormal'],
        class_mode = 'sparse',
        color_mode = 'grayscale',
        target_size = (128,128),
        batch_size = 896,
        save_to_dir = './Data_images/image_test_2_gun/Enhanced_image',
        save_prefix  = 'Enhanced image')
    return image_test_generator

x = trainGenerator()
x.next()
'''
if __name__ == '__main__':
    n = 10
    plt.figure(2,figsize=(20, 4))
    x = test_image()
    c,v = x.next()
    for i in range(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(c[i].reshape(128,128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
'''

