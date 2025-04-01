# step 1: download library 
import torch 
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class main(object):
    def __init__(self, **kwargs): 
        #use **kwargs: to capture any keyword arguments (key, values) pairs not explitly defined in its parameter list
        self.batchsize = kwargs.pop("batch_size",128)
        # step2: load the data
        # download, pre-processing + data augentament, split, create data loader
         
        
        #transform setting
        mytransform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(), 
                                        #convert image to PIL-python image library 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                        ])
        #download
        train = datasets.CIFAR10(train=True, download=True, transform=mytransform)
        test = datasets.CIFAR10(train=False, download=True, transform=mytransform)

        #split 
        train_idx, valid_idx = train_test_split(train_size=0.8, test_size=0.2,shuffle=True)
        train, val = Dataset(train_idx), Dataset(valid_idx)
        train_loader = DataLoader(train,
                                  batchsize = self.batchsize, shuffle = True)
        val_loader = DataLoader(val, batch_size=self.batchsize)
        test_loader = DataLoader(test, batchsize = self.batchsize)

        classed = ['bird', 'cat', 'deer', 'dog']

    def show_img(img):
        #we normalize image to tensor (C x H x W), now we need to convert it back (H W C)
    
        img = img *0.5 + 0.5
        img = np.transpose(img, (1,2,0))
        pass
        

    def train(self):
        pass

    def evaluate(self, epoch):
        pass
