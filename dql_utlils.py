# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:39:31 2020

@author: mjkam
"""
import os,shutil
import configparser as cfg 
import torch
import math
from datetime import datetime


def parse_name(filepath):
    RUNFILE = os.path.basename(filepath)
    RUNSTEM = os.path.splitext(RUNFILE)[0]

    config = cfg.ConfigParser()
    config.read('config_'+RUNSTEM+'.ini')
    return RUNFILE, RUNSTEM


def check_cuda(requestCuda=True):
    """
    Check if CUDA is available, set and return the device

    Inputs:
        USE_CUDA: use Cuda if True and Cuda is available, otherwise use cpu
    
    Returns:
        the device selected ('cpu' or the cuda device)
    """
    print('Requested CUDA:',requestCuda)
    if requestCuda==True:
        print('There are {} CUDA Devices installed on this system:'.format(torch.cuda.device_count()))
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print('CUDA Device:',device,torch.cuda.get_device_name(device))
        print('Use CUDA? ',use_cuda)
    else:
        use_cuda = False
        device = torch.device('cpu')
        print('Using CPU as torch device')    
    return device

def calculate_epsilon(steps_done,
                      EGREEDY_EPSILON = 0.9,
                      EGREEDY_EPSILON_FINAL = 0.02,
                      EGREEDY_DECAY = 1000):
    """
    Decays eplison with increasing steps

    Input:
        steps_done (int) : number of steps completed

    Returns:
        int - decayed epsilon
    """
    epsilon = EGREEDY_EPSILON * math.exp(-1. * steps_done / EGREEDY_DECAY )
    return max(epsilon,EGREEDY_EPSILON_FINAL)

def runprep(RUNFILE,RUNSTEM,symbol=''):
    """
    Create Unique Run Name and output directory

    Returns RUNFILE, RUNSTEM, RUNNAME, RUNPATH
    """
    RUNNAME = RUNSTEM+'_'+symbol+'_'+str(datetime.now().strftime('%Y%m%d_%H%M'))
    RUNPATH = 'results/'+RUNNAME
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(RUNPATH):
        os.makedirs(RUNPATH)  
    
    shutil.copy(RUNFILE,RUNPATH)
    shutil.copy('config_'+RUNSTEM+'.ini',RUNPATH)    
    return RUNNAME,RUNPATH

def bellman_expansion(gamma,N,r):
    g = g=[gamma**i for i in range(N+1)]
    return [r*sum(g[:i]) for i in range(N+1)]
    
    