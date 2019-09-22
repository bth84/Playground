import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

CUDA = torch.cuda.is_available()
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 10

#dimensionality of latent variable Z
ZDIMS = 20

os.chdir('./torchoutput')

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

kwargs = {'num_workers':1, 'pin_memory': True} if CUDA else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **kwargs
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **kwargs
)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #Encoder
        self.fc1 = nn.Linear(784, 400)
        self.relu = nn.ReLU()

        self.fc21 = nn.Linear(400, ZDIMS) #mu layer
        self.fc22 = nn.Linear(400, ZDIMS) #logvariance layer

        #Decoder
        self.fc3 = nn.Linear(ZDIMS, 400)
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()

    #Encoder
    def encode(self, x: Variable) -> (Variable, Variable):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    #Reparametrization trick to enable backpropagation through
    #the entire AutoEncoder Network
    def reparametrize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)

        else:
            return mu

    def decode(self, z: Variable) -> Variable:
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
if CUDA:
    model.cuda()

def loss_function(recon_x, x, mu, logvar) -> Variable:
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    #Kullback–Leibler divergence
    KLD = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * 784

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr = 1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    # toggle model to test / inference mode
    model.eval()
    test_loss = 0

    # each data is of BATCH_SIZE (default 128) samples
    for i, (data, _) in enumerate(test_loader):
        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda()

        # we're only going to infer, so no autograd at all required: volatile=True
        with torch.no_grad():
            data = Variable(data)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data.item()
            if i == 0:
              n = min(data.size(0), 8)
              # for the first 128 batch of the epoch, show the first 8 input digits
              # with right below them the reconstructed output digits
              comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
              save_image(comparison.data.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)

    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
    # digits in latent space
    sample = Variable(torch.randn(64, ZDIMS))
    if CUDA:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()

    # save out as an 8x8 matrix of MNIST digits
    # this will give you a visual idea of how well latent space can generate things
    # that look like digits
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')