from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import augmentations
augmentations.IMAGE_SIZE = 224

class clothing_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_samples=0): 
        
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}     
        self.num_samples = num_samples
        
        self.crop = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2,1.)),                    
            transforms.RandomHorizontalFlip(),                         
        ])
        self.to_tensor = transforms.Compose([                
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),        
        ])                     
        with open('%s/noisy_label_kv.txt'%self.root_dir,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root_dir+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/clean_label_kv.txt'%self.root_dir,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root_dir+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])   

        if mode == 'train': 
            self.train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root_dir,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root_dir+l[7:]
                    self.train_imgs.append(img_path)                                               
        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root_dir,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root_dir+l[7:]
                    self.test_imgs.append(img_path)            
        elif mode=='val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root_dir,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root_dir+l[7:]
                    self.val_imgs.append(img_path)
                                       
    def sample_subset(self, num_class=14):#sample a class-balanced subset
        random.shuffle(self.train_imgs)
        class_num = torch.zeros(num_class)
        self.train_imgs_subset = []
        for impath in self.train_imgs:
            label = self.train_labels[impath] 
            if class_num[label]<(self.num_samples/14) and len(self.train_imgs_subset)<self.num_samples:
                self.train_imgs_subset.append(impath)
                class_num[label]+=1
        random.shuffle(self.train_imgs_subset)    
        return    
    
    
    def __getitem__(self, index):  
        if self.mode=='train':
            img_path = self.train_imgs_subset[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            img_aug = self.crop(image)
            img_aug = augmentations.aug(img_aug,self.to_tensor)           
            return img, target, index, img_aug         
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs_subset)            
        
        
class clothing_dataloader():  
    def __init__(self, root_dir, batch_size, num_workers, num_batches=1000):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.num_batches = num_batches
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])     
        
    def run(self):        
        train_dataset = clothing_dataset(root_dir = self.root_dir,transform=self.transform_train, mode='train',num_samples=self.num_batches*self.batch_size)
        train_dataset.sample_subset()
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)   
        
        test_dataset = clothing_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test')      
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)               

        eval_dataset = clothing_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='train',num_samples=self.num_batches*self.batch_size)      
        eval_loader = DataLoader(
            dataset=eval_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)  
        return train_loader,eval_loader,test_loader