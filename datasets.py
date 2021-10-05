# Dataloaders for CUB Birds, Stanford Dogs, Foodx datasets

from __future__ import print_function, division

#import numpy as np # ?

import torch
import torch.nn.functional as F
#import torch.nn as nn # ?
#import torch.optim as optim # ?
#from torch.optim import lr_scheduler # ?

import torchvision
import torchvision.transforms as T
#from torchvision import datasets, models, transforms # ?

from PIL import Image #for food dateset

import scipy.io #for dogs dateset



#import matplotlib.pyplot as plt # ?

#import pandas as pd # ?

#import time # ?
#import os # ?
#import copy # ?



## CUB-200-2011 (Birds) Dataset:

class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content



## Stanford Dogs Dataset

class DOGDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for DOG Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}splits/file_list.mat")
        image_files = [o[0][0] for o in image_info]
        
        split_info = self.get_file_content(f"{image_root_path}/splits/{split}_list.mat")
        split_files = [o[0][0] for o in split_info]
        self.split_info = {}
        if split == 'train' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "1"
                else:
                    self.split_info[image] = "0"
        elif split== 'test' :
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "0"
                else:
                    self.split_info[image] = "1"
                    
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(DOGDataset, self).__init__(root=f"{image_root_path}Images", is_valid_file = self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        content =  scipy.io.loadmat(file_path)
        return content['file_list']



## FoodX-251 Dataset

class FOODDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, *args, **kwargs):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        # return (
        #     torchvision.transforms.functional.to_tensor(Image.open(row["path"])), row['label']
        # )
        #print(row["path"])
    
        img = Image.open(row["path"])
        img2 = img.resize((224,224), resample=0)
        #img2.save('/home/u20020067/Downloads/1.jpg')

        out = T.ToTensor()(img2)
        #print(out.shape)

        #out = F.interpolate(img, size=224)  #The resize operation on tensor.
        #print(out)

        #x = self.transform(Image.open(row["path"]))
        #print(out)
        # x = F.interpolate(x, (224, 224))

        #x = torchvision.transforms.functional.to_tensor(x)
  
        return out, row['label']

    # transform = T.Compose([
    #     T.ToTensor(),
    #     T.ToPILImage(),
    #     T.Resize(224),
    #     T.ToTensor()])



## FoodX-251 (old):

# class FOODDataset(torch.utils.data.Dataset):
#     def __init__(self, dataframe, *args, **kwargs):
#         self.dataframe = dataframe

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         # return (
#         #     torchvision.transforms.functional.to_tensor(Image.open(row["path"])), row['label']
#         # )
#         #print(row["path"])
    
#         img = Image.open(row["path"])
#         img2 = img.resize((224,224), resample=0)
#         #img2.save('/home/u20020067/Downloads/1.jpg')

#         out = T.ToTensor()(img2)
#         #print(out.shape)

#         #out = F.interpolate(img, size=224)  #The resize operation on tensor.
#         #print(out)

#         #x = self.transform(Image.open(row["path"]))
#         #print(out)
#         # x = F.interpolate(x, (224, 224))

#         #x = torchvision.transforms.functional.to_tensor(x)
  
#         return out, row['label']

#     # transform = T.Compose([
#     #     T.ToTensor(),
#     #     T.ToPILImage(),
#     #     T.Resize(224),
#     #     T.ToTensor()])