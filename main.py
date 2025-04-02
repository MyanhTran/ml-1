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
        self.batchsize = kwargs.pop("batch_size", 16)
        # step2: load the data
        # download, pre-processing + data augentament, split, create data loader
         
        
        #transform setting
        mytransform = transforms.Compose([
                                        transforms.Resize((256,256)), #images in Flower102 doesn't have same size
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(degrees=10), 
                                        #convert image to PIL-python image library 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                        ])
        #download
        train = datasets.Flowers102(root='./data', split='train', download=True, transform=mytransform)
        val = datasets.Flowers102(root='./data', split='val', download=True, transform=mytransform)
        test = datasets.Flowers102(root='./data', split='test', download=True, transform=mytransform)


        self.train_loader = DataLoader(train, batch_size=self.batchsize, shuffle = True)
        self.val_loader = DataLoader(val, batch_size=self.batchsize, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=self.batchsize, shuffle=False)

    def show_batch_img(self):
        #get batch image from 1 single interation of data loader
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)

        nrows = int(np.sqrt(self.batchsize))
        ncols = int(np.ceil(self.batchsize / nrows))

        #display the grid 
        fig, axes = plt.subplots(nrows, ncols, figsize=(12,12))
        axes = axes.flatten()

        for img, label, ax in zip(images, labels, axes):
            img = img.permute((1,2,0)).numpy()
            ax.imshow(img)
            ax.set_title(f"label: {label.item()}")
            ax.axis('off')
        plt.show()

    def train(self):
        pass

    def evaluate(self, epoch):
        pass

main_object = main()
main_object.show_batch_img()