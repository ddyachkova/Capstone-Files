#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import sys
import numpy as np
import glob
import time
import glob


# In[2]:



# In[16]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *

import torch_resnet_single_2_jets_7_layers as networks


from pylab import rcParams
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import pyarrow as pa
import pyarrow.parquet as pq
import h5py

from functools import partial
import argparse

import energyflow
from energyflow.emd import emd
# In[4]:


from ResNetVAE_Modules import * 


# In[5]:


os.environ["CUDA_VISIBLE_DEVICES"]=str(0)


# In[14]:


def get_args():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-total_samples', '--total_samples' , default = 150000, type=int, help='Total number of samples to run the traning on')
    parser.add_argument('-batch_size', '--batch_size', type=int, default = 500, help='Batch size')
    parser.add_argument('-random_sampler', '--random_sampler' , default = True, type=bool, help='Shuffle the samples in DataLoader')
    parser.add_argument('-num_epochs', '--num_epochs' , default = 1, type=int, help='Number of epochs')
    parser.add_argument('-num_blocks', '--num_blocks' , default = 3, type=int, help='Number of blocks in the ResNet')
    parser.add_argument('-name', '--name', default = 'ECAL', type=str, help='Which channel(s) to run the training on')

    return parser.parse_args()


if __name__ == "__main__":
    wdir = r'C:\Users\Darya\Documents\GitHub\Autoencoders'
    expt_name = 'ResNet_VAE_Jets_all_channels'

    os.chdir(wdir)
    args = get_args()
    for d in ['MODELS', 'METRICS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))
    num_files = int(args.total_samples/(32*1000))
    os.chdir(wdir + '\Data\Parquet_Data')
    max_ind = max([int(f.split('-')[1].split('.')[0])  for f in os.listdir(wdir + '\Data\Parquet_Data') if (f.split('-')[1].split('.')[3] == 'parquet')])
    datasets = ['jets_hdf5_X_ecal_hcal_tracks-%i.h5.snappy.parquet'%i for i in range(num_files+1)]
    train_cut = int(0.8 * args.total_samples)
    train_loader, val_loader = train_val_loader(datasets, train_cut, args.batch_size, random_sampler = args.random_sampler)
    print ('Loaded Data')
    train(train_loader, val_loader, args.num_blocks, args.num_epochs, args.name, args.batch_size, wdir, expt_name, imgs = False)

