# step 1: download library 
import torch 
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.models import resnet50
import torch.nn as nn 
import time

class main(object):
    def __init__(self, **kwargs): 
        #use **kwargs: to capture any keyword arguments (key, values) pairs not explitly defined in its parameter list
        self.batchsize = kwargs.pop("batch_size", 36)
        self.lr = kwargs.pop("learning_rate", 0.01)
        self.beta = kwargs.pop("beta", 0.999)
        self.epochs = kwargs.pop("epochs", 1)

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

        #initialize loss function 
        self.model = resnet50()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params= self.model.parameters(),
                                          lr=self.lr)

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
        
        #1 loop over mini batches 
        for epoch in range(self.epochs):
            #2 set up for training 
            self.train_loss = 0.0
            self.val_loss = 0.0
            correct = 0
            self.model.train()
            for idx, (data, target) in enumerate(self.train_loader):
                start = time.time()
                self.optimizer.zero_grad() #reset gradients
                #3 forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                #4 backward pass: compute gradient
                loss.backward()
                #5 update parameters
                self.optimizer.step() #update the paramter for each epoch
                #6 track loss and accuracy 
                import pdb
                pdb.set_trace()
                self.train_loss += loss.item() #item() extract scalar from tensor
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()

        self.train_loss = self.train_loss / len(self.train_loader.dataset)
        train_acc = (correct / len(self.train_loader.dataset)) * 100 

        print(start, self.train_loss, train_acc)
        

    def evaluate(self, epoch):
        pass

main_object = main()
main_object.train()