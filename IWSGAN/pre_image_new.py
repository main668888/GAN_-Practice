# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:40:56 2021

@author: user
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
from glob import glob
import os
from PIL import Image
nbofdata = 3000
nbofdata_i=1
images=[]
size = (128,128)
def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized

for folders in glob("all_Data_images_2/knife"):
    print(folders)
    nbofdata_i=1
    for filename in os.listdir(folders):
        print(filename)
        if nbofdata_i <= nbofdata:
            img =cv2.imread(os.path.join(folders, filename))
            print('imgae load shape',img.shape)
            img = crop_square(img,128)
            # img=cv2.resize(img,size,cv2.INTER_NEAREST)
            print('resize 之後',img.shape)
            img = cv2.medianBlur(img, 3)
            ret,th1 = cv2.threshold(img,125,255,cv2.THRESH_BINARY)
            if img is not None:
                images.append(th1)
                
            cv2.imwrite('./image_pre_knife/'+filename, th1)
            nbofdata_i+=1
            

# img = cv2.imread('B0044_0028.png',0)
# img_small = imutils.resize(img,width = 128)
# img_small = cv2.medianBlur(img_small, 3)
# ret,th1 = cv2.threshold(img_small,25,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img_small,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img_small, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)

# size = 128
# img = cv2.imread('B0044_0028.png',0)
# resized_1 = resize2SquareKeepingAspectRation(img, size, cv2.INTER_AREA)
# resized_2 = crop_square(img, size)
# titles = ['resize2SquareKeeping','crop_square']
# images = [resized_1,resized_2]

# for i in range(2):
#     plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()













