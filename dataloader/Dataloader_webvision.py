from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
from PIL import ImageFile
import augmentations
augmentations.IMAGE_SIZE = 299
ImageFile.LOAD_TRUNCATED_IMAGES = True


class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root+str(c))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(c),img)])                
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_class, transform_strong=None): 
        self.root = root_dir
        self.transform = transform
        self.transform_strong = transform_strong
        self.mode = mode  
        
        self.crop = transforms.Compose([
            transforms.RandomResizedCrop(299, scale=(0.4,1.)),                    
            transforms.RandomHorizontalFlip(),                         
        ])
        self.to_tensor = transforms.Compose([                
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),            
        ])        
             
        if self.mode=='test':
            self.val_imgs = []
            self.val_labels = {}            
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target<num_class:
                        self.val_imgs.append(img)
                        self.val_labels[img]=target                             
        else:    
            self.train_imgs = []
            self.train_labels = {}            
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target<num_class:
                        self.train_imgs.append(img)
                        self.train_labels[img]=target            
   
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            img_aug = self.crop(image)
            img_aug = augmentations.aug(img_aug,self.to_tensor)           
            return img, target, index, img_aug  
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


class webvision_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir

        self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(299, scale=(0.4,1.)), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(320), 
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])         

    def run(self):

        train_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="train", num_class=self.num_class)                
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)                                                

        test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', num_class=self.num_class)      
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)               

        eval_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='train', num_class=self.num_class)      
        eval_loader = DataLoader(
            dataset=eval_dataset, 
            batch_size=self.batch_size*2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)               

        imagenet_val = imagenet_dataset(root_dir=self.root_dir, transform=self.transform_imagenet, num_class=self.num_class)      
        imagenet_loader = DataLoader(
            dataset=imagenet_val, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)        
        
        return train_loader,eval_loader,test_loader,imagenet_loader     

