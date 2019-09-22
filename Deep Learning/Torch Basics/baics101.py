import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import sys
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.out = nn.Linear(200, 10)
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.out(x), dim=1)
        return x


model = Classifier()
if use_cuda:
    model.cuda()


transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

traindata = torchvision.datasets.MNIST('../../data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=2**8, shuffle=True)

testdata = torchvision.datasets.MNIST('../../data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=2**8, shuffle=True)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 30
losses = []
test_losses = []
accuracies = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

for e in range(epochs):

    num_batches = len(trainloader)
    running_loss = 0

    for i, (images, label) in enumerate(trainloader):
        sys.stdout.write('\rBatch {}/{} - Epoch {:02d}/{}'.format(i+1, num_batches, e+1, epochs))
        sys.stdout.flush()

        if use_cuda:
            images = images.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            model.eval()

            for images, labels in testloader:
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                out = model(images)
                test_loss += criterion(out, labels)

                ps = torch.exp(out)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()
        test_losses.append(test_loss/len(testloader))
        accuracies.append(accuracy/len(testloader))

    sys.stdout.write(' - Loss: {:.4f}\n'.format(running_loss/num_batches))
    sys.stdout.flush()

    losses.append(running_loss/num_batches)

    ax1.clear()
    ax2.clear()
    ax1.plot(range(len(losses)), losses)
    ax1.plot(range(len(test_losses)), test_losses)
    ax1.legend(['Train', 'Test'])
    ax2.plot(range(len(accuracies)), accuracies)
    plt.draw()
    plt.pause(.001)

print('Finis')
plt.show()
