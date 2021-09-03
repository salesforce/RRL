from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import augmentations
augmentations.IMAGE_SIZE = 32

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, ratio, noise_mode, root_dir, transform, mode, noise_file=''): 
        
        self.ratio = ratio # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        
        self.crop = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),                       
        ])
        self.to_tensor = transforms.Compose([                
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),          
        ])    
        
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.images = test_dic['data']
                self.images = self.images.reshape((10000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))  
                self.labels = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.images = test_dic['data']
                self.images = self.images.reshape((10000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))  
                self.labels = test_dic['fine_labels']                            
        else:    
            self.images=[]
            clean_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    self.images.append(data_dic['data'])
                    clean_label = clean_label+data_dic['labels']
                self.images = np.concatenate(self.images)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                self.images = train_dic['data']
                clean_label = train_dic['fine_labels']
            self.images = self.images.reshape((50000, 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))
            self.clean_label = np.array(clean_label)
            
            if os.path.exists(noise_file):
                self.labels = json.load(open(noise_file,"r"))
            else:    #inject label noise   
                self.labels = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.ratio*50000)            
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            self.labels.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[clean_label[i]]
                            self.labels.append(noiselabel)                    
                    else:    
                        self.labels.append(clean_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(self.labels,open(noise_file,"w"))       
                
            #self.images = np.load('noise_file/cifar10_train_corrupt.npy')  #for corrupted cifar-10
                
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
    def __init__(self, dataset, ratio, noise_mode, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.ratio = ratio
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        if self.dataset=='cifar10':
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
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self):
        train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, ratio=self.ratio, root_dir=self.root_dir, transform=self.transform_train, mode="train",noise_file=self.noise_file)      
        
        eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, ratio=self.ratio, root_dir=self.root_dir, transform=self.transform_test, mode="eval",noise_file=self.noise_file) 
        
        test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, ratio=self.ratio, root_dir=self.root_dir, transform=self.transform_test, mode="test") 
    
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