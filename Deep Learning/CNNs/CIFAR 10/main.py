import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #convolutional layer, sees 32x32x3 image tensor
        #has 3 input channels (R,G,B Colors), goes for 16 filters and
        #has a kernel size of 3. To avoid shrinking of the output layers
        #padding is set to 1, since kernel size is 3
        self.conv1 = nn.Conv2d(3,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        #final classifier
        #linear layer #1:  64 * 4 * 4, 500
        self.fc1 = nn.Linear(64*4*4, 500)
        #output to 10 different classes
        self.fc2 = nn.Linear(500,10)

        #Regularization
        self.dropout = nn.Dropout(.25)

    def forward(self,x):

        #Sequence of Convolutionals
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        #flatten image input
        x = x.view(-1, 64 * 4 * 4)

        #add dropout
        x = self.dropout(x)

        #classifier
        #both classifier layers with relu activation and first one with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x