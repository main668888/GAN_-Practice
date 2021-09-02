# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:40:08 2020

@author: user
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
x = torch.linspace(-5, 5, 100) #creat array from -5 to 5
print(x)

x = Variable(x) #charnge to variable
#-------------------------------------
x_np = x.data.numpy()
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
#--------------------------------------
plt.figure(1, figsize=(10, 10)) #創一個figure，並設定寬度、長度
plt.subplot(221) #2*2表格的第一格
plt.plot(x_np, y_relu,'-b', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')
plt.subplot(222)
plt.plot(x_np, y_sigmoid,'-g', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
plt.subplot(223)
plt.plot(x_np, y_tanh,'-r', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
plt.subplot(224)
plt.plot(x_np, y_softplus,'-c', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')
plt.show()