# step 1: download library 
import torch 
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class main(object):
    def __init__(self, **kwargs): 
        #use **kwargs: to capture any keyword arguments (key, values) pairs not explitly defined in its parameter list
        self.batchsize = kwargs.pop("batch_size",16)
        # step2: load the data
        # download, pre-processing + data augentament, split, create data loader
         
        
        #transform setting
        mytransform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(degrees=10), 
                                        #convert image to PIL-python image library 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                        ])
        #download
        train = datasets.CIFAR10(root='./data', train=True, download=True, transform=mytransform)
        test = datasets.CIFAR10(root='./data', train=False, download=True, transform=mytransform)

        #split 
        train_idx, valid_idx = train_test_split(range(len(train)), test_size=0.2,shuffle=True)
        
        train_subset = Subset(train, train_idx)
        val_subset = Subset(train, valid_idx)

        self.train_loader = DataLoader(train_subset, batch_size=self.batchsize, shuffle = True)
        self.val_loader = DataLoader(val_subset, batch_size=self.batchsize, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=self.batchsize, shuffle=False)

        classed = ['bird', 'cat', 'deer', 'dog']

    def show_img(self, img):
        img = img *0.5 + 0.5 #we normalize image to tensor (C x H x W), now we need to convert it back (H W C)
        img = np.transpose(img.numpy(), (1,2,0))
        plt.imshow(img)
        plt.axis('off')
        plt.show()        

    def train(self):
        pass

    def evaluate(self, epoch):
        pass

#create an instance of a class 
main_object = main()
#access the train loader (this is why we use reference 'self' in main)
train_loader = main_object.train_loader

#get a batch of image 
data_iter = iter(train_loader) #iter convert Iterable object (DataLoader) object to iterable
images, labels = next(data_iter) #next give items in iterable

#show batch of images 
img_grid = make_grid(images, nrow=4)
main_object.show_img(img_grid)