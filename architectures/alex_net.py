import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os,sys,inspect
import cv2
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import common.dataloader
from common.constants import *

INPUT_SIZE = 128

#TRAINSET_DIR = "./uniform_trainset_diff"
# TRAINSET_DIR = "./uniform_trainset"
# TESTSET_DIR = "./testset"
# FAKE_DIR = TRAINSET_DIR + "/Fake"
# TRUE_DIR = TRAINSET_DIR + "/True"
# GAN_DIR  = TRAINSET_DIR + "/gan"
#idk why the import was not working

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10#30
batch_size = 128
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
# Optimize function needs nn.Module instance on construction, hence callback function is used.
optimize_getter = lambda m, m_lr: optim.SGD(m.parameters(), lr=m_lr, momentum=0.9, weight_decay=0.0005)

# Dataset
train_dataset = torchvision.datasets.ImageFolder( root=TRAINSET_DIR, transform=torchvision.transforms.ToTensor())
#test_dataset = torchvision.datasets.ImageFolder( root=TESTSET_DIR, transform=torchvision.transforms.ToTensor())
test_dataset = common.dataloader.TestDataSet(root=TESTSET_DIR, transform=torchvision.transforms.ToTensor())
#test40_dataset = torchvision.datasets.ImageFolder( root=TESTSET40_DIR, transform=torchvision.transforms.ToTensor())

#train_dataset = torchvision.datasets.ImageFolder( root=TRAINSET_DIR, transform=torchvision.transforms.ToTensor())
#test_dataset = torchvision.datasets.ImageFolder( root=TRAINSET_DIR, transform=torchvision.transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#test40_loader = torch.utils.data.DataLoader(dataset=test40_dataset, batch_size=1, shuffle=False, num_workers=4)


#more params
"""
Stucture
CONV1: in_channels=3, out_channels=96, kernel_size=12, stride=2, padding=0, bias=True
MAXPOOL1: kernel_size=3, stride=2
NORM1
CONV2: in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True
MAXPOOL2: kernel_size=3, stride=2
NORM2
CONV3: in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True
CONV4: in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True
CONV5: in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True
MaxPOOL3: kernel_size=4, stride=2
FC6: 9216, 4096
FC7: 4096, 1000
FC8: 1000, 2
"""

# Resting Fake Face Model
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=12, stride=2, padding=0, bias=True)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        #self.batchnorm1 = nn.BatchNorm2d(96)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        #self.batchnorm2 = nn.BatchNorm2d(256)
        
        #add in 3 more conv layers
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.max_pool3 = torch.nn.MaxPool2d(kernel_size=4, stride=2)

        self.fc1 = torch.nn.Linear(9216, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 2)

        # self.fc1 = torch.nn.Linear(16*30*30, 4200)
        # self.fc2 = torch.nn.Linear(4200, 800)
        # self.fc3 = torch.nn.Linear(800, 3)

        #torch.save(self, ckpt_path)

    def forward(self, x):
        #round one of conv-max-norm
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        #x = self.batchnorm1(x)

        #round two of conv-max-norm
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        #x = self.batchnorm2(x)

        #three rounds of conv
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool3(x)

        # Flatten
        #x = x.view(-1, 16*30*30)
        x = x.view(-1, 9216)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x

    def train(self, start_epoch=0):
        ckpt_path_base = CKPT_DIR + "/alex_net"

        optimizer = optimize_getter(self, learning_rate)

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
                outputs = self(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # If L1 loss function is applied, do one hot encoding.
                if type(loss_func) == type(nn.L1Loss()):
                    labels = F.one_hot(labels, num_classes=10)
                    labels = labels.type(torch.FloatTensor).to(device)
                loss = loss_func(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 10 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch+1, num_epochs, i+1, total_step, loss.item(), 100*correct/total))
                    total = 0
                    correct = 0
            #if (epoch + 1) % 10 == 0:
            torch.save(self, ckpt_path_base + f"_{epoch+1}.ckpt")
            print("Saved epoch: {}".format(epoch+1))


    def test(self):
        #self = torch.load("checkpoints/resnet101_50.ckpt")
        # Test the self
        self.eval()

        # Output text string
        output_str = str()

        correct = 0
        total = 0
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = self(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for output, path in list(zip(outputs, paths)):
                output_str += os.path.basename(path) + ",{:.7f}".format(output[1])
                output_str += "\n"

        print(f'Accuracy of the network on the {total} test images: {100 * correct / total} %')

        f = open("grader/AUROC/prediction_file.txt", "w+")
        f.write(output_str.rstrip("\n"))
        f.close()


    def test40(self):
        output = ""
        trans = transforms.Compose([torchvision.transforms.ToTensor()])
        with torch.no_grad():
            for filename in os.listdir(TESTSET40_DIR):
                img = cv2.imread(os.path.join(TESTSET40_DIR, filename))
                if img is not None:
                    img_tensor = trans(img) #torch.from_numpy(img)
                    img_tensor = img_tensor.view((1, 3, 128, 128))
                    #image = img_tensor.to(device)
                    outputs = self(img_tensor)

                    if (filename != "0001.jpg"):
                        output += "\n"
                    
                    output += filename + ",{:.7f}".format((outputs[0][1]).item())

        f = open("output.txt","w+")
        f.write(output)
        f.close()


if __name__ == "__main__":
    latest_checkpoint = 0
    test40 = True

    if (latest_checkpoint == 0):
        alex = AlexNet().to(device)
        alex.train()
        alex.test()
    else:
        ckpt_path = CKPT_DIR + "/alex_net" + f"_{latest_checkpoint}.ckpt"
        alex = torch.load(ckpt_path).to(device)
        if (latest_checkpoint < num_epochs):
            alex.train(start_epoch=latest_checkpoint)
        if (test40):
            alex.test40()
        else:
            alex.test()
