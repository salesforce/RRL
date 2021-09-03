from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import augmentations
from torchvision import datasets
from pdb import set_trace as breakpoint

augmentations.IMAGE_SIZE = 32

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, ratio, open_noise, root_dir, transform, mode, noise_file=''): 
        
        self.ratio = ratio # close set noise ratio
        self.open_ratio = 0.4 # open set noise ratio 
        self.transform = transform
        self.mode = mode  
        
        self.crop = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),                       
        ])
        self.to_tensor = transforms.Compose([                
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),          
        ])    
                       
        if self.mode=='test':            
            test_dic = unpickle('%s/cifar-10/test_batch'%root_dir)
            self.images = test_dic['data']
            self.images = self.images.reshape((10000, 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))  
            self.labels = test_dic['labels']                       
        else:    
            self.images=[]
            clean_label=[]
            for n in range(1,6):
                dpath = '%s/cifar-10/data_batch_%d'%(root_dir,n)
                data_dic = unpickle(dpath)
                self.images.append(data_dic['data'])
                clean_label = clean_label+data_dic['labels']
            self.images = np.concatenate(self.images)            
            
            #inject open set data
            num_open_noise = int(self.open_ratio*50000)
            if open_noise=='cifar100':
                cifar100_dic = unpickle('%s/cifar-100/train'%root_dir)
                cifar100_data = cifar100_dic['data']
                cifar100_idx = list(range(50000))
                random.shuffle(cifar100_idx)                   
                randidx_cifar100 = cifar100_idx[:num_open_noise]   
                open_noise_data = cifar100_data[randidx_cifar100]
                
            elif open_noise=='svhn': 
                svhn_dataset = datasets.SVHN('%s/svhn'%root_dir, split='train', download=True)
                svhn_data = svhn_dataset.data
                svhn_idx = list(range(svhn_data.shape[0]))
                random.shuffle(svhn_idx)        
                randidx_svhn = svhn_idx[:num_open_noise]           
                svhn_data = svhn_data.reshape((svhn_data.shape[0], 3*32*32))
                open_noise_data = svhn_data[randidx_svhn]                                               
            
            self.images = np.concatenate([self.images, open_noise_data])                    
            self.images = self.images.reshape((self.images.shape[0], 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))
            
            self.clean_label = np.array(clean_label,dtype=int)
            self.clean_label = np.concatenate([self.clean_label,-np.ones(num_open_noise,dtype=int)])
            
            if os.path.exists(noise_file):
                self.labels = json.load(open(noise_file,"r"))
                self.labels = np.array(self.labels,dtype=int)
            else:    
                #inject label noise
                self.labels = np.array(self.clean_label,dtype=int)
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.ratio*50000)            
                noise_idx = idx[:num_noise]
                self.labels[noise_idx] = np.random.randint(0,9,num_noise)               
                self.labels[50000:] = np.random.randint(0,9,num_open_noise)      

                print("save noisy labels to %s ..."%noise_file)        
                json.dump(self.labels.tolist(),open(noise_file,"w"))       
       
                
    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        if self.mode=='train':
            img = Image.fromarray(img)
            img_orig = self.transform(img)                
            img_aug = self.crop(img)
            img_aug = augmentations.aug(img_aug,self.to_tensor)          
            return img_orig, target, index, img_aug
        else:
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        return len(self.images)         
        
        
class cifar_dataloader():  
    def __init__(self, ratio, open_noise, batch_size, num_workers, root_dir, noise_file=''):
        self.ratio = ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.open_noise = open_noise
        
        self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])    

    def run(self):
        train_dataset = cifar_dataset(ratio=self.ratio, open_noise=self.open_noise, root_dir=self.root_dir, transform=self.transform_train, mode="train",noise_file=self.noise_file)      
        
        eval_dataset = cifar_dataset(ratio=self.ratio, open_noise=self.open_noise, root_dir=self.root_dir, transform=self.transform_test, mode="eval",noise_file=self.noise_file) 
        
        test_dataset = cifar_dataset(ratio=self.ratio, open_noise=self.open_noise, root_dir=self.root_dir, transform=self.transform_test, mode="test") 
    
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)        
        eval_loader = DataLoader(
            dataset=eval_dataset, 
            batch_size=self.batch_size*4,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)  
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size*4,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)          
        return train_loader,eval_loader,test_loader