"""Define a generic class for training and testing learning algorithms."""
from __future__ import print_function
import os
import os.path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

import utils
import datetime
import logging

from architectures.PreResNet import *
from architectures.InceptionResNetV2 import *
from architectures.resnet import *

import tensorboard_logger as tb_logger

import faiss
import numpy as np
import torchnet

class Algorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir'])
        self.set_log_file_handler()
        self.logger.info('Algorithm options %s' % opt)
        self.opt = opt
        self.init_all_networks()
        self.init_all_criterions()
        self.curr_epoch = 0
        self.optimizers = {}
        self.acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
        self.checkpoint_dir = opt['checkpoint_dir']       

    def set_experiment_dir(self,directory_path):
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)
    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)
        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)
        now_str = datetime.datetime.now().__str__().replace(' ','_')

        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)        
        self.tb_logger = tb_logger.Logger(logdir=os.path.join(log_dir,'tensorboard'), flush_secs=2)
                                          
    def init_all_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}
        for key, val in networks_defs.items():
            self.logger.info('Set network %s' % key)
            name = val['name']
            net_opt = val['opt']
            self.optim_params[key] = val['optim_params'] if ('optim_params' in val) else None
            pretrained = val['pretrained'] if ('pretrained' in val) else None
            self.networks[key] = self.init_network(name, net_opt, pretrained, key)

    def init_network(self, name, net_opt, pretrained, key):        
        if name == 'resnet50':
            network = nn.DataParallel(resnet50(num_class=self.opt['data_train_opt']['num_class'],low_dim=self.opt['data_train_opt']['low_dim'],pretrained=pretrained))       
        elif name == 'inception':   
            network = nn.DataParallel(InceptionResNetV2(num_classes=self.opt['data_train_opt']['num_class'],low_dim=self.opt['data_train_opt']['low_dim']))  
        if name == 'presnet':
            network = PreResNet18(num_class=self.opt['data_train_opt']['num_class'],low_dim=self.opt['data_train_opt']['low_dim'])                   
        return network


    def init_all_optimizers(self):
        self.optimizers = {}
        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams != None:
                self.optimizers[key] = self.init_optimizer(
                        self.networks[key], oparams, key)

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.logger.info('Initialize optimizer: %s with params: %s for network: %s'
            % (optim_type, optim_opts, key))
        if optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                momentum=optim_opts['momentum'],
                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
                weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)
        return optimizer

    def init_all_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            if 'weight' in crit_opt.keys():
                crit_opt['weight'] = torch.load(self.opt['data_train_opt']['class_weight'])           
            self.criterions[key] = self.init_criterion(crit_type, crit_opt)
    
    def init_criterion(self, ctype, copt):
        return getattr(nn, ctype)(**copt)

    def load_to_gpu(self):
        print('loading models to gpu')
        for key, net in self.networks.items():
            self.networks[key] = net.cuda()
        for key, criterion in self.criterions.items():
            self.criterions[key] = criterion.cuda()

    def save_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            self.save_network(key, epoch, suffix=suffix)
            if self.optimizers[key] != None:             
                self.save_optimizer(key, epoch, suffix=suffix)

    def load_checkpoint(self, epoch, train=True, suffix=''):
        for key, net in self.networks.items(): # Load networks
            self.load_network(key, epoch, suffix)

        if train: # initialize and load optimizers
            self.init_all_optimizers()
            for key, net in self.networks.items():
                if self.optim_params[key] == None: continue
                self.load_optimizer(key, epoch,suffix)

        self.curr_epoch = epoch

    def save_network(self, net_key, epoch, suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(self.exp_dir, net_key, epoch)+suffix
        state = {'epoch': epoch,'network': self.networks[net_key].state_dict()}
        torch.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(self.exp_dir, net_key, epoch)+suffix
        state = {'epoch': epoch,'optimizer': self.optimizers[net_key].state_dict()}
        torch.save(state, filename)

    def load_network(self, net_key, epoch,suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(self.checkpoint_dir, net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        self.logger.info('Load checkpoint from %s' % (filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.networks[net_key].load_state_dict(checkpoint['network'])

    def load_optimizer(self, net_key, epoch,suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(self.checkpoint_dir, net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        self.logger.info('Load optimizer from %s' % (filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])

    def _get_net_checkpoint_filename(self, file_dir, net_key, epoch):
        return os.path.join(file_dir, net_key+'_net_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, file_dir, net_key, epoch):
        return os.path.join(file_dir, net_key+'_optim_epoch'+str(epoch))

            
    def solve(self, data_loader_train, data_loader_eval, test_loader, imagenet_loader=None):
        
        self.max_num_epochs = self.opt['max_num_epochs']  
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers['model'], T_max=self.max_num_epochs-start_epoch, eta_min=0.0002)     
        #**********************************************************       
        if self.curr_epoch==0:
            self.run_train_warmup(data_loader_train,self.curr_epoch) #warm-up for several iterations to initalize prototypes    
            
        train_stats = {}    
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):      
            self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch+1, self.max_num_epochs))
            if 'clothing' in self.opt['dataset']:
                data_loader_eval.dataset.sample_subset()
                data_loader_train.dataset.train_imgs_subset = data_loader_eval.dataset.train_imgs_subset

            features,labels,probs = self.compute_features(data_loader_eval, self.networks['model'], len(data_loader_eval.dataset))  
                
            self.prototypes = [] 
            if self.opt['knn'] and self.curr_epoch>=self.opt['knn_start_epoch']:                
                if self.curr_epoch==self.opt['knn_start_epoch'] or 'clothing' in self.opt['dataset']:     
                    #initalize the soft label as model's softmax prediction
                    gt_score = probs[labels>=0,labels]
                    gt_clean = gt_score>self.opt['low_th'] 
                    self.soft_labels=probs.clone() 
                    self.soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), self.opt['data_train_opt']['num_class']).scatter_(1, labels[gt_clean].view(-1,1), 1)     
                    
                if 'cifar' in self.opt['dataset']:
                    #generate new soft label, we can analysis the label correction performance with cifar
                    self.label_clean(features,labels,probs,data_loader_eval.dataset.clean_label)    
                    self.test_knn(self.networks['model'], test_loader, features)
                else:
                    #generate new soft label 
                    self.label_clean(features,labels,probs)
                    
                    self.test_knn(self.networks['model'], test_loader, features)
                    #self.test_webvision_knn(self.networks['model'], features, test_loader, imagenet_loader)
                
                features = features[self.clean_idx]   
                pseudo_labels = self.hard_labels[self.clean_idx]   
                for c in range(self.opt['data_train_opt']['num_class']):      
                    prototype = features[np.where(pseudo_labels.numpy()==c)].mean(0)    #compute prototypes with pseudo-label
                    self.prototypes.append(torch.Tensor(prototype))                                                     
            else:   
                for c in range(self.opt['data_train_opt']['num_class']): 
                    prototype = features[np.where(labels.numpy()==c)].mean(0)    #compute prototypes as mean embeddings       
                    self.prototypes.append(torch.Tensor(prototype))                                
            self.prototypes = torch.stack(self.prototypes).cuda()
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)    #normalize the prototypes                
           
            if self.opt['data_train_opt']['ramp_epoch']: #ramp up the weights for prototypical loss (optional)
                self.w_proto = min(1+self.curr_epoch*(self.opt['data_train_opt']['w_proto']-1)/self.opt['data_train_opt']['ramp_epoch'],self.opt['data_train_opt']['w_proto'])
            else:
                self.w_proto = self.opt['data_train_opt']['w_proto']    
            self.logger.info('==> Set to w_proto = %.2f' % (self.w_proto))
            
            #perform training for 1 epoch
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            for k,v in train_stats.items():
                self.tb_logger.log_value(k,v,self.curr_epoch)
            
            # save model checkpoint every 10 epochs
            if (self.curr_epoch+1)%10==0:
                self.save_checkpoint(self.curr_epoch+1) 
            
            if imagenet_loader:    
                self.test_webvision(self.networks['model'], test_loader, imagenet_loader)
            else:
                self.test(self.networks['model'], test_loader)
                

                
    def run_train_warmup(self, data_loader, epoch):
        self.logger.info('Warm-up Training: %s' % os.path.basename(self.exp_dir))                
        self.networks['model'].train()
        disp_step   = self.opt['disp_step'] if ('disp_step' in self.opt) else 50
        train_stats = utils.DAverageMeter()    
        for idx, batch in enumerate(tqdm(data_loader)):        
            if idx>self.opt['data_train_opt']['warmup_iters']:
                break            
            train_stats_this = self.train_step(batch,True)
            train_stats.update(train_stats_this)
            if (idx+1) % disp_step == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' % (epoch+1, idx+1, len(data_loader), train_stats.average()))   
        return train_stats.average()
    
    
    def run_train_epoch(self, data_loader, epoch):      
        self.logger.info('Training: %s' % os.path.basename(self.exp_dir))       
        self.networks['model'].train()
        disp_step   = self.opt['disp_step'] if ('disp_step' in self.opt) else 50
        train_stats = utils.DAverageMeter()
               
        self.scheduler.step()
        for param_group in self.optimizers['model'].param_groups:
            lr = (param_group['lr'])
            self.logger.info('==> Set to optimizer lr = %.4f' % (lr))
            break            
        for idx, batch in enumerate(tqdm(data_loader)):
            train_stats_this = self.train_step(batch,False)
            train_stats.update(train_stats_this)
            if (idx+1) % disp_step == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' % (epoch+1, idx+1, len(data_loader), train_stats.average()))
        return train_stats.average()

    def compute_features(self, dataloader, model, N):
        print('Compute features')
        model.eval()
        batch_size = dataloader.batch_size
        for i, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                inputs = batch[0].cuda(non_blocking=True)
                output, feat = model(inputs)             
                feat = feat.data.cpu().numpy()
                prob = F.softmax(output,dim=1)
                prob = prob.data.cpu()
            if i == 0:
                features = np.zeros((N, feat.shape[1]),dtype='float32')
                labels = torch.zeros(N,dtype=torch.long)                        
                probs = torch.zeros(N,self.opt['data_train_opt']['num_class']) 
            if i < len(dataloader) - 1:
                features[i * batch_size: (i + 1) * batch_size] = feat
                labels[i * batch_size: (i + 1) * batch_size] = batch[1]
                probs[i * batch_size: (i + 1) * batch_size] = prob
            else:
                # special treatment for final batch
                features[i * batch_size:] = feat
                labels[i * batch_size:] = batch[1]
                probs[i * batch_size:] = prob
        return features,labels,probs   
    
    def train_step(self, batch):
        pass

    def test_webvision(self, model, test_loader, imagenet_loader):
        with torch.no_grad():
            self.logger.info('==> Testing on Webvision...')
            self.acc_meter.reset()
            model.eval()        
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs,_ = model(inputs)    
                self.acc_meter.add(outputs,targets)
            webvision_acc = self.acc_meter.value()
            self.logger.info('Accuracy is %.2f%% (%.2f%%)'%(webvision_acc[0],webvision_acc[1]))

            self.logger.info('==> Testing on Imagenet...')
            self.acc_meter.reset()
            for batch_idx, (inputs, targets) in enumerate(imagenet_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs,_ = model(inputs)     
                self.acc_meter.add(outputs,targets)
            imagenet_acc = self.acc_meter.value()
            self.logger.info('Accuracy is %.2f%% (%.2f%%)'%(imagenet_acc[0],imagenet_acc[1]))
            
            self.tb_logger.log_value('WebVision top1 Acc', webvision_acc[0], self.curr_epoch)
            self.tb_logger.log_value('WebVision top5 Acc', webvision_acc[1], self.curr_epoch)
            self.tb_logger.log_value('ImageNet top1 Acc', imagenet_acc[0], self.curr_epoch) 
            self.tb_logger.log_value('ImageNet top5 Acc', imagenet_acc[1], self.curr_epoch)
        return   

    def test_webvision_knn(self, model, features, test_loader, imagenet_loader):
        k = self.opt['n_neighbors']      
        index = faiss.IndexFlatIP(features.shape[1])       
        index.add(features)           
        with torch.no_grad():
            self.logger.info('==> Testing on Webvision...')
            self.acc_meter.reset()
            model.eval()        
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, test_feat = model(inputs)                  
                batch_size = inputs.size(0)     
                dist = np.zeros((batch_size, k))
                neighbors = np.zeros((batch_size, k))  
                D,I = index.search(test_feat.data.cpu().numpy(),k)                  
                neighbors = torch.LongTensor(I)
                weights = torch.exp(torch.Tensor(D)/self.temperature).unsqueeze(-1)           
                score = torch.zeros(batch_size,self.opt['data_train_opt']['num_class'])
                for n in range(batch_size):           
                    neighbor_labels = self.soft_labels[neighbors[n]]
                    score[n] = (neighbor_labels*weights[n]).sum(0)                                           
                self.acc_meter.add(score,targets)
            webvision_acc = self.acc_meter.value()
            self.logger.info('Accuracy is %.2f%% (%.2f%%)'%(webvision_acc[0],webvision_acc[1]))

            self.logger.info('==> Testing on Imagenet...')
            self.acc_meter.reset()
            for batch_idx, (inputs, targets) in enumerate(imagenet_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, test_feat = model(inputs)                  
                batch_size = inputs.size(0)     
                dist = np.zeros((batch_size, k))
                neighbors = np.zeros((batch_size, k))  
                D,I = index.search(test_feat.data.cpu().numpy(),k)                  
                neighbors = torch.LongTensor(I)
                weights = torch.exp(torch.Tensor(D)/self.temperature).unsqueeze(-1)           
                score = torch.zeros(batch_size,self.opt['data_train_opt']['num_class'])
                for n in range(batch_size):           
                    neighbor_labels = self.soft_labels[neighbors[n]]
                    score[n] = (neighbor_labels*weights[n]).sum(0)                                            
                self.acc_meter.add(score,targets)
            imagenet_acc = self.acc_meter.value()
            self.logger.info('Accuracy is %.2f%% (%.2f%%)'%(imagenet_acc[0],imagenet_acc[1]))

            self.tb_logger.log_value('WebVision knn top1 Acc', webvision_acc[0], self.curr_epoch)
            self.tb_logger.log_value('WebVision knn top5 Acc', webvision_acc[1], self.curr_epoch)
            self.tb_logger.log_value('ImageNet knn top1 Acc', imagenet_acc[0], self.curr_epoch) 
            self.tb_logger.log_value('ImageNet knn top5 Acc', imagenet_acc[1], self.curr_epoch)
        return   
    
    def test(self, model, test_loader):
        with torch.no_grad():
            self.logger.info('==> Testing...')
            self.acc_meter.reset()
            model.eval()        
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs,_ = model(inputs)    
                self.acc_meter.add(outputs,targets)
            accuracy = self.acc_meter.value()
            self.logger.info('Accuracy is %.2f%%'%(accuracy[0]))
            self.tb_logger.log_value('Accuracy', accuracy[0], self.curr_epoch)
        return 
    
    def test_knn(self, model, test_loader, features):          
        k = self.opt['n_neighbors']      
        index = faiss.IndexFlatIP(features.shape[1])       
        index.add(features)        
        with torch.no_grad():
            self.logger.info('==> Knn Testing...')
            self.acc_meter.reset()
            model.eval()        
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, test_feat = model(inputs)                  
                batch_size = inputs.size(0)     
                dist = np.zeros((batch_size, k))
                neighbors = np.zeros((batch_size, k))  
                D,I = index.search(test_feat.data.cpu().numpy(),k)                  
                neighbors = torch.LongTensor(I)
                weights = torch.exp(torch.Tensor(D)/self.temperature).unsqueeze(-1)           
                score = torch.zeros(batch_size,self.opt['data_train_opt']['num_class'])
                for n in range(batch_size):           
                    neighbor_labels = self.soft_labels[neighbors[n]]
                    score[n] = (neighbor_labels*weights[n]).sum(0)                
                self.acc_meter.add(score,targets)
            accuracy = self.acc_meter.value()
            self.logger.info('Knn Accuracy is %.2f%%'%(accuracy[0]))
            self.tb_logger.log_value('Knn Accuracy', accuracy[0], self.curr_epoch-1)
        return  
    
    
    def label_clean(self,features,labels,probs,clean_label=None):     
        self.logger.info('==> Knn Searching...')
        # initalize knn search
        N = features.shape[0]      
        k = self.opt['n_neighbors']      
        index = faiss.IndexFlatIP(features.shape[1])   
        
        index.add(features)  
        D,I = index.search(features,k+1)  
        neighbors = torch.LongTensor(I) #find k nearest neighbors excluding itself
        
        score = torch.zeros(N,self.opt['data_train_opt']['num_class']) #holds the score from weighted-knn
        weights = torch.exp(torch.Tensor(D[:,1:])/self.temperature)  #weight is calculated by embeddings' similarity
        for n in range(N):           
            neighbor_labels = self.soft_labels[neighbors[n,1:]]
            score[n] = (neighbor_labels*weights[n].unsqueeze(-1)).sum(0)  #aggregate soft labels from neighbors
        self.soft_labels = (score/score.sum(1).unsqueeze(-1) + probs)/2  #combine with model's prediction as the new soft labels
        
        #consider the ground-truth label as clean if the soft label outputs a score higher than the threshold
        gt_score = self.soft_labels[labels>=0,labels]
        gt_clean = gt_score>self.opt['low_th'] 
        self.soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), self.opt['data_train_opt']['num_class']).scatter_(1, labels[gt_clean].view(-1,1), 1)  
        
        #get the hard pseudo label and the clean subset used to calculate supervised loss
        max_score,self.hard_labels = torch.max(self.soft_labels,1)  
        self.clean_idx = max_score>self.opt['high_th']            
        self.logger.info('Number of clean samples: %d'%self.clean_idx.sum())
        self.tb_logger.log_value('num_clean',self.clean_idx.sum(),self.curr_epoch) 
        
        if clean_label is not None: #statistics for cifar
            clean_label = torch.LongTensor(clean_label)
            self.acc_meter.reset()
            self.acc_meter.add(self.soft_labels[:50000],clean_label[:50000])
            self.logger.info('Correction accuracy:%.2f%%'%self.acc_meter.value()[0])
            self.tb_logger.log_value('Correction accuracy',self.acc_meter.value()[0],self.curr_epoch)      

            effective_noise = 1-1.0*(self.hard_labels[self.clean_idx]==clean_label[self.clean_idx]).sum()/self.clean_idx.sum()
            self.logger.info('Effective noise ratio:%.2f%%'%effective_noise)
            self.tb_logger.log_value('effective_noise',effective_noise,self.curr_epoch)   
            
        else:  #statistics for webvision
            num_correction = (self.hard_labels[self.clean_idx]!=labels[self.clean_idx])
            self.logger.info('Number of corrected labels: %d'%num_correction.sum())
            self.tb_logger.log_value('num_correction',num_correction.sum(),self.curr_epoch)   
            save_dir = os.path.join(self.exp_dir,'data')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.save(save_dir+'/clean_idx_%d.npy'%self.curr_epoch,self.clean_idx)
            np.save(save_dir+'/pseudo_label_%d.npy'%self.curr_epoch,self.hard_labels)
        return   
    
    