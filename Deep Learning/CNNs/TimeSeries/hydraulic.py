import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

label = pd.read_csv('C:/datasets/ConditionMonitoringHydraulic/profile.txt', sep='\t')
data = ['TS1.txt','TS2.txt','TS3.txt','TS4.txt']
df = pd.DataFrame()

#read and concat
for txt in data:
    read_df = pd.read_csv('C:/datasets/ConditionMonitoringHydraulic/'+txt, sep='\t', header=None)
    df = df.append(read_df)

df.reset_index()
df.index = list(range(len(df)))

#scale data
def scale(df):
    return (df - df.mean(axis=0))/df.std(axis=0)
df = df.apply(scale)

df.loc(axis=1)[0].plot()
#plt.show()

class CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(CNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers

        self.lstm1 = nn.LSTM(input_size, hidden_size, n_layers)
        self.lstm2 = nn.LSTM(hidden_size, 1)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hn):

        x, hn = self.lstm1(x, hn, self.hidden_size)
        x = x.view(-1, self.hidden_size)
        x = self.fc(x)
        return x, hn

class myData(Dataset):
    def __init__(self, df):
        self.df = df
        df.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        return torch.Tensor(self.df.iloc[ix].values)

trainLoader = DataLoader(myData(df), batch_size=64, shuffle=False)

model = CNN(1, 1, 32, 1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

for e in range(10):
    hidden = None
    losses = 0
    for ix, data in enumerate(trainLoader):
        print(ix+1, len(trainLoader))
        inputs = data[:-1]
        labels = data[1:]
        optimizer.zero_grad()

        print(inputs.shape)
        plt.plot(inputs.numpy()[:])
        plt.show()
        out, hidden = model(inputs, hidden)
        hidden = hidden.data
        loss = criterion(inputs, labels)
        loss.backward()
        optimizer.step()

        losses += loss.item()
    else:
        print(losses)