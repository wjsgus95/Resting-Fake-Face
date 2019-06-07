import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import os
import cv2

from common.constants import *

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 30
batch_size = 16
learning_rate = 0.004
loss_func = nn.CrossEntropyLoss()
# Optimize function needs nn.Module instance on construction, hence callback function is used.
optimize_getter = lambda m, m_lr: optim.Adam(m.parameters(), lr=m_lr)

train_transform = transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(),
                                      transforms.RandomRotation(degrees=15),
                                      torchvision.transforms.ToTensor()])

# Dataset
train_dataset = torchvision.datasets.ImageFolder(root=TRAINSET_DIR, transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(root=TESTSET_DIR, transform=torchvision.transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

def train(model, start_epoch=0):
    ckpt_path_base = CKPT_DIR + "/resnet50"

    optimizer = optimize_getter(model, learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):

            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # If L1 loss function is applied, do one hot encoding.
            if type(loss_func) == type(nn.L1Loss()):
                labels = F.one_hot(labels, num_classes=2)
                labels = labels.type(torch.FloatTensor).to(device)
            loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch+1, num_epochs, loss.item(), 100*correct/total))
        total = 0
        correct = 0
        torch.save(model, ckpt_path_base + f"_{epoch+1}.ckpt")

def test(model):
    # Test the model
    model.eval()

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total} %')

def test40(model):
    #produce a batch first
    trans = transforms.Compose([torchvision.transforms.ToTensor()])
    images = []
    for filename in os.listdir(TESTSET40_DIR):
        img = cv2.imread(os.path.join(TESTSET40_DIR, filename))
        if img is not None:
            img_tensor = trans(img)
            images.append(img_tensor)

    batch = torch.stack(images, dim=0).to(device)

    #run model
    outputs = model(batch)

    #produce output
    output = ""
    for img_num, prec in enumerate(outputs[:, 1], 1):
        if (img_num != 1):
            output += "\n"
                
        filename = "{:04d}.jpg".format(img_num)
        output += filename + ",{:.7f}".format(prec.item())
               
    f = open("output_sn.txt","w+")
    f.write(output)
    f.close()

def find_latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
        return 0

    ep = 30
    while(not os.path.exists(CKPT_DIR + "/resnet50" + f"_{ep}.ckpt")):
        ep -= 1
        if (ep <= 0):
            break
    return ep

        
