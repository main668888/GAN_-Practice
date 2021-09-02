# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:24:29 2020

@author: user
"""

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder,self).__init__()
        self.ngpu = ngpu

        assert 16 % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential()
        
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        
        for t in range(n_extra_layers):
            main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))
            
        while csize > 4 :
            in_feat = cndf
            out_feat = cndf*2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat,out_feat),
                            nn.Conv2d(in_feat,out_feat,4,2,1,bias = False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
            
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main
            
    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output
    
class Decoder(nn.Module):
    def __init__ (self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        
            