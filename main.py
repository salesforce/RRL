'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
import torch
import numpy
import random

parser = argparse.ArgumentParser()
parser.add_argument('--exp',         type=str, required=True, default='',  help='config file with parameters of the experiment')
parser.add_argument('--checkpoint',  type=int,      default=0,     help='checkpoint (epoch id) that will be loaded')
parser.add_argument('--num_workers', type=int,      default=8,    help='number of data loading workers')
parser.add_argument('--cuda'  ,      type=bool,     default=True,  help='enables cuda')
parser.add_argument('--disp_step',   type=int,      default=50,    help='display step during training')
parser.add_argument('--seed',        type=int,      default=123,    help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
numpy.random.seed(args.seed)
random.seed(args.seed)
    
exp_config_file = os.path.join('.','config',args.exp+'.py')

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = os.path.join('.',config['exp_directory']) # the place where logs, models, and other stuff will be stored
config['checkpoint_dir'] = os.path.join('.',config['checkpoint_dir']) 
print("Loading experiment %s from file: %s" % (args.exp, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']

config['disp_step'] = args.disp_step
algorithm = getattr(alg, config['algorithm_type'])(config)
if args.cuda: # enable cuda
    algorithm.load_to_gpu()
if args.checkpoint > 0: # load checkpoint
    algorithm.load_checkpoint(args.checkpoint, train=True)

print('create data loader')
if config['dataset']=='webvision':
    import dataloader.Dataloader_webvision as dataloader
    loader = dataloader.webvision_dataloader(batch_size=data_train_opt['batch_size'],num_workers=args.num_workers,\
                                             root_dir=config['data_path'],num_class = data_train_opt['num_class'])             
    train_loader,eval_loader,test_loader,imagenet_loader = loader.run()    
    algorithm.solve(train_loader,eval_loader,test_loader,imagenet_loader)
elif 'cifar' in config['dataset']:
    if config['openset']:
        import dataloader.Dataloader_cifar_openset as dataloader
        loader = dataloader.cifar_dataloader(batch_size=data_train_opt['batch_size'],num_workers=args.num_workers,\
                                             ratio=config['noise_ratio'],root_dir=config['data_path'],\
                                             noise_file = config['noise_file'],open_noise=config['openset'])         
    else:       
        import dataloader.Dataloader_cifar as dataloader
        loader = dataloader.cifar_dataloader(batch_size=data_train_opt['batch_size'],num_workers=args.num_workers,\
                                             ratio=config['noise_ratio'],noise_mode = config['noise_mode'],\
                                             root_dir=config['data_path'],noise_file = config['noise_file'],\
                                             dataset = config['dataset'])                     
    train_loader,eval_loader,test_loader = loader.run()    
    algorithm.solve(train_loader,eval_loader,test_loader)    
elif 'clothing' in config['dataset']:   
    import dataloader.Dataloader_clothing1m as dataloader
    loader = dataloader.clothing_dataloader(batch_size=data_train_opt['batch_size'],num_workers=args.num_workers,        
                                            root_dir=config['data_path'])                     
    train_loader,eval_loader,test_loader = loader.run()    
    algorithm.solve(train_loader,eval_loader,test_loader)      
