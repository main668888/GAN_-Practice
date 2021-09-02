# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:04:51 2020

@author: user
"""

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=int, default=7)
parser.add_argument('--mode', type=str, default='test', help='train, test')
args = parser.parse_args()

(X_tra)()