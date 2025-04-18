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
from models.VanillaModel import VanillaModel

class Avg_helper(object):
    "helper function to compute average time in encapsulated manner"
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum = val * n 
        self.avg = self.sum / self.count



class main(object):
    def __init__(self, **kwargs): 
        #use **kwargs: to capture any keyword arguments (key, values) pairs not explitly defined in its parameter list
        self.batchsize = kwargs.pop("batch_size", 36)
        self.lr = kwargs.pop("learning_rate", 0.01)
        self.beta = kwargs.pop("beta", 0.999)
        self.epochs = kwargs.pop("epochs", 1)
        self.model_type = kwargs.pop('model', 'VanillaModel')

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
        
        #select model 
        if self.model_type == 'VanillaModel':
            self.model = VanillaModel()
            print(type(VanillaModel))

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
        for epoch in range(self.epochs):
            #train loop
            self.train_step(epoch)
            #eval loop
            self._evaluate(epoch)
        pass

    def train_step(self, epoch):
        iter_time = Avg_helper()
        loss_helper = Avg_helper()
        acc_helper = Avg_helper()
        
        self.model.train()
        
        #run batches 
        for index, (data, target) in enumerate(self.train_loader):
            start = time.time()

            #forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            #zero_grad() set grad to None for memory efficiency and advoid accumualting grad from prev batch
            self.optimizer.zero_grad()

            #backward pass
            loss.backward()
            self.optimizer.step()
            
            #get accuracy 
            batch_acc = self._get_accuracy(output, target)
            batch_size = output.shape[0]
            #update loss and acc into the accumulating loss, acc of epoch
            loss_helper.update(loss.item(), batch_size)
            acc_helper.update(batch_acc, batch_size)
            iter_time.update(time.time() - start)

            batch_number = len(self.train_loader)
            if index%10 == 0:
                print(
                    (
                        "epoch: {0},{1}/{2} \t"
                        "time: {iter_time.val:.2f} {iter_time.avg:.2f} \t"
                        "loss: {loss.val:2f} {loss.avg:.2f} \t"
                        "acc: {acc.val:.2f} {acc.avg:.2f} \t"
                    ).format(
                        epoch, index, batch_number,
                        iter_time=iter_time, 
                        loss=loss_helper,
                        acc=acc_helper
                    )
                )


    def _get_accuracy(self, ouput, target):
        batch_size = target.shape[0]
        #get the max prob of the tensor, with
        _, prediction = torch.max(ouput, dim=-1)
        #.eq() compare predictions and target 
        correct = prediction.eq(target).sum() * 1.0
        return correct / batch_size

    def _evaluate(self, epoch):
        iter_time = Avg_helper()
        loss_helper = Avg_helper()
        acc_helper = Avg_helper()
        
        self.model.eval()
        for index, (data, target) in enumerate(self.val_loader):
            start = time.time()

            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)

            batch_acc = self._get_accuracy(output, target)
            batch_size = output.shape[0]
            batch_number = len(self.val_loader)

            loss_helper.update(loss.item(), batch_size)
            acc_helper.update(batch_acc, batch_size)

            iter_time = time.time() - start
            if index%10:
                print(
                    (
                        'epoch: {0}, {1}/{2} \t'
                        'time: {iter_time.val:.2g} {iter_time.avg:.2f} \t'
                        'loss: {loss.val:.2f} ]f'
                        'acc: {acc.val:.2f}'
                    ).format(
                        epoch, index, batch_number,
                        iter_time = iter_time,
                        loss=loss_helper,
                        acc=acc_helper
                    )
                )
        return acc.avg



main_object = main()
main_object.train()