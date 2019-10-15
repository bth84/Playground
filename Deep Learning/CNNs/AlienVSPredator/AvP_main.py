import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import mlflow




# ---------------
# data generators
# ---------------
normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
}

image_datasets = {
    'train': datasets.ImageFolder('data/train', transform=data_transforms['train']),
    'validation': datasets.ImageFolder('data/validation', transform=data_transforms['validation'])
}

data_loaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'],
                                         batch_size=32,
                                         shuffle=True,
                                         num_workers=4),
    'validation': torch.utils.data.DataLoader(image_datasets['validation'],
                                         batch_size=32,
                                         shuffle=True,
                                         num_workers=4)
}




# ------------------
# create the network
# ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048,128),
    nn.ReLU(inplace=True),
    nn.Linear(128,2)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.fc.parameters())


# ---------------
# train the model
# ---------------
def train_model(model, criterion, optimizer, num_epochs=3):
    loss = 10.
    acc = 0.
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss.item(),
                                                        epoch_acc.item()))
            loss = min(loss, epoch_loss.item())
            acc = max(acc, epoch_acc.item())

    mlflow.log_param("Epochs", num_epochs)
    mlflow.log_metric("Loss", loss)
    mlflow.log_metric("Accuracy", acc)
    return model



if __name__ == '__main__':
    model_trained = train_model(model, criterion, optimizer, num_epochs=5)

    # -----------------------
    # save and load the model
    # -----------------------
    torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')
    model = models.resnet50(pretrained=False).to(device)
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)).to(device)
    model.load_state_dict(torch.load('models/pytorch/weights.h5'))

    # ---------------------------------
    # predictions on sample test images
    # ---------------------------------

    # validation_img_paths = ["data/validation/alien/24.jpg",
    #                         "data/validation/alien/27.jpg",
    #                         "data/validation/predator/33.jpg"]

    humans = ['data/humans/adnan.jpg',
              'data/humans/ben.jpg',
              'data/humans/ela.jpg']

    #img_list = [Image.open(img_path) for img_path in validation_img_paths]
    img_list = [Image.open(img_path) for img_path in humans]
    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                    for img in img_list])

    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100 * pred_probs[i, 0],
                                                              100 * pred_probs[i, 1]))
        ax.imshow(img)

    fig.savefig('images/output/predicts.png')
    mlflow.log_artifact('images/output/predicts.png')
    plt.show()
