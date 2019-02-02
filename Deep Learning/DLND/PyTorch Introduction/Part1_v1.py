import torch

def activation(x):
    """
    Sigmoid activation function

    :param x:
    :type x: torch.Tensor
    :return:
    :rtype:
    """
    return 1/(1 + torch.exp(-x))

#set seed
torch.manual_seed(7)

# features are 5 random normal variables
features = torch.randn((1,5))

# true weights for our data, random normal variables again
weights = torch.randn_like(features)

# and a true bias term
bias = torch.randn((1,1))

y = activation(torch.sum(features * weights) + bias)
#y = activation((features * weights).sum() + bias)

#doing the matrix multiplication via torch.mm
# print(activation(torch.mm(features,weights.view(5,1)) + bias))
# print(y)


#_____________________________________________________#

#define the size of each layer in our network
n_input = features.shape[1]
n_hidden = 2
n_output = 1

#weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

f = activation

y = f(torch.mm(f(torch.mm(features, W1) + B1), W2) + B2)

#print(y)


#_____________________________________________________#
import numpy as np

a = np.random.rand(4,3)
b = torch.from_numpy(a)
#print(b.numpy())

#Multiply PyTorch Tensor by 2, in place
#print(b.mul_(2))

#Numpy array matches new values from tensor
#print(a)



#_____________________________________________________#
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,  .5, .5), (.5, .5, .5))
])

trainset = datasets.MNIST('mnist_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# print(type(images))
# print(images.shape)
# print(labels.shape)

#plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
#plt.show()

def activation(x):
    return 1 / (1 + torch.exp(-x))

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1,1)


#flatten the input images
inputs = images.view(images.shape[0], -1)

#create parameters
w1 = torch.randn(784,256)
b1 = torch.randn(256)

w2 = torch.randn(256,10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)
out = torch.mm(h, w2) + b2
probabilities = softmax(out)

#print(probabilities.shape)
#print(probabilities.sum(dim=1))



#_______________real nn Module__________________________#
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)

        #Output layer, 10 units - one for each unit
        self.output = nn.Linear(256,10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #Pass the input tensor through each of our operations

        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

model = Network()
#print(model)



#_______cleaner code in torch_______#
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(784,256)
        self.output = nn.Linear(256,10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)

        return x

model = Network()
#print(model)


#_______ReLU + complexity_______#
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.Relu(self.hidden2(x))
        x = F.softmax(output)

        return x

model = Network()
#print(model)


#_______Training_______#

#Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
])

#Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True,
                          transform=transform,
                          train=True,
                          )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Build the Feedforward Network
model = nn.Sequential(
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,64),
    nn.ReLU(),
    nn.Linear(64,10)
)

#Define the loss
criterion = nn.CrossEntropyLoss()

#Get our data
images, labels = next(iter(trainloader))
#flatten images
images = images.view(images.shape[0], -1)

#Forward pass, get our logits
logits = model(images)

#Calculate the loss with the logits and labels
loss = criterion(logits, labels)

#print(loss)



#_______Exercise_______#
#Use log softmax as output activation function

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256,64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.LogSoftmax(dim=1)
)
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
logits = model(images)
loss = criterion(logits, labels)
#print(loss)


#_____Optimizer_____#
from torch import optim

optimizer = optim.SGD(model.parameters(), lr=.01)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256,64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.LogSoftmax(dim=1)
)
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

#clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()
#logits = model(images)

#Forward pass, then backward pass, then update weights
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()

#print('Gradient -', model[0].weight.grad)

#Take an update step and few the new weights
optimizer.step()
#print('Updated weights -', model[0].weight)



#____ Epochs_____#
#Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True,
                          transform=transform,
                          train=True,
                          )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256,64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.LogSoftmax(dim=1)
)
optimizer = optim.SGD(model.parameters(), lr=.003)
criterion = nn.NLLLoss()

epochs = 5
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#         #flatten images into 784 long vectors
#         images = images.view(images.shape[0], -1)
#
#         #zero_grad!
#         optimizer.zero_grad()
#
#         output = model.forward(images)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#     else:
#         print(f"Training loss: {running_loss/len(trainloader)}")


# images, labels = next(iter(trainloader))
# img = images[0].view(1, 784)
#
# with torch.no_grad():
#     logits = model.forward(img)
#
# ps = F.softmax(logits, dim=1)
#plt.imshow(img.view(28,28))
#plt.show()
#print(ps)

#_____Fashion Mnist______#

trainset = datasets.FashionMNIST('~/.pytorch/MNIST_data/', download=True, transform=transform, train=True,)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=.2)

    def forward(self, x):
        #make sure, tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = .003)

epochs = 30
step = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        logps = model(images)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        print(f"Training loss: {running_loss}")

        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(test_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}..".format(e+1, epochs),
              "Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
              "Test loss: {:.3f}..".format(test_loss/len(testloader)),
              "Test accuracy: {:.3f}".format(accuracy/len(testloader)))