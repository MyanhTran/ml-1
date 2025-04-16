import torch 
import torch.nn as nn 

class VanillaModel(nn.Module):
    #Module is a base class for all nn 
    #so our VanillaModel should also be a subclass of this Module class
    def __init__(self):
        super().__init__()
        #input shape: 3,256,256
        # (H - K) / S + 1
        self.convu1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1) 
        #output shape: (256-3)/1 +1= (20,254,254)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #ouput shape: 254 // 2: (20,127,127)
        self.convu2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1)
        #output shape: (127-3)/1 +1 = (20,125,125)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #output shape: 125//2: (20,62,62)
        self.fc = nn.Linear(20*62*62, 102)
        test

    def forward(self, x):
        x = nn.functional.relu(self.convu1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.convu2(x))
        x = self.pool2(x)
        x = torch.flatten(x,1)
        return self.fc(x)