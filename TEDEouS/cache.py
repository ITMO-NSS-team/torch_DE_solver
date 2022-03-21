# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:50:12 2021

@author: user
"""
import pickle
import datetime
import torch
import os 
import glob
import numpy as np
from TEDEouS.metrics import point_sort_shift_loss


def save_model(model,state,optimizer_state,cache_dir='../cache/',name=None):
    if name==None:
        name=str(datetime.datetime.now().timestamp())
    if os.path.isdir(cache_dir):
        torch.save({'model':model, 'model_state_dict': state,
                'optimizer_state_dict': optimizer_state}, cache_dir+name+'.tar')
    else:
        os.mkdir(cache_dir)
        torch.save({'model':model, 'model_state_dict': state,
                'optimizer_state_dict': optimizer_state}, cache_dir+name+'.tar')
    return

def cache_lookup(prepared_grid, operator, bconds, lambda_bound=0.001,cache_dir='../cache/',nmodels=None,verbose=False,norm=None): 
    # looking for all *.tar files in directory cache_dir
    files=glob.glob(cache_dir+'*.tar')
    # if files not found
    if len(files)==0:
        best_checkpoint=None
        min_loss=np.inf
        return best_checkpoint, min_loss
    # at some point we may want to reduce the number of models that are
    # checked for the best in the cache
    if nmodels==None:
        # here we take all files that are in cache
        cache_n=np.arange(len(files))
    else:
        # here we take random nmodels from the cache
        cache_n=np.random.choice(len(files), nmodels, replace=False)
    cache_same_architecture=[]
    min_loss=np.inf
    best_model=0
    best_checkpoint={}
    for i in cache_n:
        file=files[i]
        checkpoint = torch.load(file)
        model=checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        # this one for the input shape fix if needed
        # it is taken from the grid shape
        if model[0].in_features!=prepared_grid.shape[-1]:
            continue
        #     model[0]=torch.nn.Linear(prepared_grid.shape[-1],model[0].out_features)
        # model.eval()
        l=point_sort_shift_loss(model, prepared_grid, operator, bconds, lambda_bound=lambda_bound,norm=norm)      
        if l<min_loss:
            min_loss=l
            best_checkpoint['model']=model
            best_checkpoint['model_state_dict']=model.state_dict()
            best_checkpoint['optimizer_state_dict']=checkpoint['optimizer_state_dict']
            if verbose:
                print('best_model_num={} , loss={}'.format(i,l))
    if best_checkpoint=={}:
        best_checkpoint=None
        min_loss=np.inf
    return best_checkpoint,min_loss
        

def cache_retrain(model,cache_checkpoint,grid,verbose=False):
    # do nothing if cache is empty
    if cache_checkpoint==None:
        optimizer_state=None
        return model,optimizer_state
    # if models have the same structure use the cache model state
    if str(cache_checkpoint['model']) == str(model):
        model=cache_checkpoint['model']
        model.load_state_dict(cache_checkpoint['model_state_dict'])
        model.eval()
        optimizer_state=cache_checkpoint['optimizer_state_dict']
        if verbose:
            print('Using model from cache')
    # else retrain the input model using the cache model 
    else:
        optimizer_state=None
        model_state=None
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        cache_model=cache_checkpoint['model']
        cache_model.load_state_dict(cache_checkpoint['model_state_dict'])
        cache_model.eval()
        loss = torch.mean(torch.square(cache_model(grid)-model(grid)))
        t=1
        def closure():
            optimizer.zero_grad()
            loss = torch.mean((cache_model(grid)-model(grid))**2)
            loss.backward()
            return loss
        t=1
        while loss>1e-5 and t<1e5:
            optimizer.step(closure)
            loss = torch.mean(torch.square(cache_model(grid)-model(grid)))
            t+=1
            if verbose:
                print('Retrain from cache t={}, loss={}'.format(t,loss))
    return model, optimizer_state
