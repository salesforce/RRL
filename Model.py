from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
import os
import torchnet as tnt
import utils
import pickle
from tqdm import tqdm
import time
import numpy as np

from random import shuffle,sample
from . import Algorithm
from pdb import set_trace as breakpoint

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Model(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.temperature = self.opt['data_train_opt']['temperature']              
        self.alpha = self.opt['data_train_opt']['alpha']
        self.w_inst = self.opt['data_train_opt']['w_inst']
        self.w_recon = self.opt['data_train_opt']['w_recon']
        
    def train_step(self, batch, warmup):       
        if self.opt['knn'] and self.curr_epoch>=self.opt['knn_start_epoch'] and not warmup:
            return self.train_pseudo(batch)
        else:   
            return self.train(batch,warmup=warmup)

    def train_naive(self, batch):
        data = batch[0].cuda(non_blocking=True)
        target = batch[1].cuda(non_blocking=True)
        record = {}
        
        output,_ = self.networks['model'](data)
        loss = self.criterions['loss'](output,target)
        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(output,target)[0].item()  
        
        self.optimizers['model'].zero_grad()
        loss.backward()            
        self.optimizers['model'].step()          
        return record    
    
    
    def train(self, batch, warmup=True):
        data = batch[0].cuda(non_blocking=True)
        target = batch[1].cuda(non_blocking=True)
        batch_size = data.size(0)
        record = {}
        
        output,feat,error_recon = self.networks['model'](data,do_recon=True)
        loss = self.criterions['loss'](output,target)
        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(output,target)[0].item()   

        loss_recon = error_recon.mean()
        loss += self.w_recon*loss_recon     
        record['loss_recon'] = loss_recon.item()
        
        if not warmup:    
            data_aug = batch[3].cuda(non_blocking=True)       

            shuffle_idx = torch.randperm(batch_size)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            feat_aug, error_recon_aug = self.networks['model'](data_aug[shuffle_idx],do_recon=True,has_out=False)  
            feat_aug = feat_aug[reverse_idx]
            
            ##**************Reconstruction loss****************
            loss_recon_aug = error_recon_aug.mean()
            loss += self.w_recon*loss_recon_aug     
            record['loss_recon'] += loss_recon_aug.item()            
            
            ##**************Instance contrastive loss****************
            sim_clean = torch.mm(feat, feat.t())
            mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
            sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

            sim_aug = torch.mm(feat, feat_aug.t())
            sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)   
            
            logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug.view(batch_size,-1,1)).squeeze(-1)
            logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

            logits = torch.cat([logits_pos,logits_neg],dim=1)
            instance_labels = torch.zeros(batch_size).long().cuda()
            
            loss_instance = self.criterions['loss_instance'](logits/self.temperature, instance_labels)                
            loss += self.w_inst*loss_instance
            record['loss_inst'] = loss_instance.item()
            record['acc_inst'] = accuracy(logits,instance_labels)[0].item()           
            
            ##**************Mixup Prototypical contrastive loss****************     
            L = np.random.beta(self.alpha, self.alpha)     
            labels = torch.zeros(batch_size, self.opt['data_train_opt']['num_class']).cuda().scatter_(1, target.view(-1,1), 1) 
            
            if 'cifar' in self.opt['dataset']:
                inputs = torch.cat([data,data_aug],dim=0)
                idx = torch.randperm(batch_size*2) 
                labels = torch.cat([labels,labels],dim=0)
            else: #do not use augmented data to save gpu memory    
                inputs = data            
                idx = torch.randperm(batch_size)              
            
            input_mix = L * inputs + (1 - L) * inputs[idx]  
            labels_mix = L * labels + (1 - L) * labels[idx]
               
            feat_mix = self.networks['model'](input_mix,has_out=False)  

            logits_proto = torch.mm(feat_mix,self.prototypes.t())/self.temperature      
            loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1))
            record['loss_proto'] = loss_proto.item()          
            loss += self.w_proto*loss_proto  
            
        self.optimizers['model'].zero_grad()
        loss.backward()            
        self.optimizers['model'].step()       
        
        return record 
    
    
    def train_pseudo(self, batch):
        data = batch[0].cuda(non_blocking=True)   
        data_aug = batch[3].cuda(non_blocking=True)    
        index = batch[2] 
        batch_size = data.size(0)
        target = self.hard_labels[index].cuda(non_blocking=True)
        clean_idx = self.clean_idx[index]    

        record = {}
        
        output,feat,error_recon = self.networks['model'](data,do_recon=True)

        loss = self.criterions['loss'](output[clean_idx],target[clean_idx])
        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(output[clean_idx],target[clean_idx])[0].item()          

        shuffle_idx = torch.randperm(batch_size)
        mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
        reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
        feat_aug, error_recon_aug = self.networks['model'](data_aug[shuffle_idx],do_recon=True,has_out=False)   
        feat_aug = feat_aug[reverse_idx]
        
        ##**************Recon loss****************
        loss_recon = error_recon.mean()+error_recon_aug.mean()
        loss += self.w_recon*loss_recon     
        record['loss_recon'] = loss_recon.item()            

        ##**************Instance contrastive loss****************
        sim_clean = torch.mm(feat, feat.t())
        mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
        sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

        sim_aug = torch.mm(feat, feat_aug.t())
        sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)   

        logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug.view(batch_size,-1,1)).squeeze(-1)
        logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

        logits = torch.cat([logits_pos,logits_neg],dim=1)
        instance_labels = torch.zeros(batch_size).long().cuda()

        loss_instance = self.criterions['loss_instance'](logits/self.temperature, instance_labels)                
        loss += self.w_inst*loss_instance
        record['loss_inst'] = loss_instance.item()
        record['acc_inst'] = accuracy(logits,instance_labels)[0].item()           

        ##**************Mixup Prototypical contrastive loss****************     
        L = np.random.beta(self.alpha, self.alpha)    
        
        labels = torch.zeros(batch_size, self.opt['data_train_opt']['num_class']).cuda().scatter_(1, target.view(-1,1), 1)  
        labels = labels[clean_idx]      
        
        if 'cifar' in self.opt['dataset']:
            inputs = torch.cat([data[clean_idx],data_aug[clean_idx]],dim=0)
            idx = torch.randperm(clean_idx.sum()*2) 
            labels = torch.cat([labels,labels],dim=0)
        else: #do not use augmented data to save gpu memory    
            inputs = data[clean_idx]            
            idx = torch.randperm(clean_idx.sum())  
             
        input_mix = L * inputs + (1 - L) * inputs[idx]  
        labels_mix = L * labels + (1 - L) * labels[idx]

        feat_mix = self.networks['model'](input_mix,has_out=False)  

        logits_proto = torch.mm(feat_mix,self.prototypes.t())/self.temperature      
        loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1))
        record['loss_proto'] = loss_proto.item()          
        loss += self.w_proto*loss_proto          
            
        self.optimizers['model'].zero_grad()
        loss.backward()            
        self.optimizers['model'].step()       
        
        return record 
        