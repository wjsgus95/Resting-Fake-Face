import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# import os,sys,inspect, cv2
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

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

# Dataset
train_dataset = torchvision.datasets.ImageFolder( root=TRAINSET_DIR, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder( root=TESTSET_DIR, transform=torchvision.transforms.ToTensor())
#train_dataset = torchvision.datasets.ImageFolder( root=PRETRAIN_DIR, transform=torchvision.transforms.ToTensor())
#test_dataset = torchvision.datasets.ImageFolder( root=TRAINSET_DIR, transform=torchvision.transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Resting Fake Face Model
class RFF(torch.nn.Module):
    def __init__(self):
        super(RFF, self).__init__()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)

        # Block 1
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)

        # Block 2
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)

        # Block 3
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True)

        # Block 4
        #self.fc1 = torch.nn.Linear(32768, 1024)
        self.fc1 = torch.nn.Linear(270848, 1024)
        self.batchnorm = nn.BatchNorm2d(1024)
        self.fc2 = torch.nn.Linear(1024, 2)
        #self.fc3 = torch.nn.Linear(1024, 2)

        #torch.save(self, ckpt_path)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.dropout(x)

        # Block 2
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.dropout(x)

        # Block 3
        x = self.conv7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Block 4
        x = x.view(-1, 270848)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc3(x)
        x = self.softmax(x)

        return x

    def train(self, start_epoch=0):
        ckpt_path_base = CKPT_DIR + "/shallownet_v3"

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

    def test(self):
        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the {total} test images: {100 * correct / total} %')

    def test40(self):
        #produce a batch first
        trans = transforms.Compose([torchvision.transforms.ToTensor()])
        images = []
        with torch.no_grad():
            for filename in os.listdir(TESTSET40_DIR):
                img = cv2.imread(os.path.join(TESTSET40_DIR, filename))
                if img is not None:
                    img_tensor = trans(img)
                    images.append(img_tensor)

        batch = torch.stack(images, dim=0)

        #run model
        outputs = self(batch)

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

# if __name__ == "__main__":
#     #rff = RFF().to(device)
#     #rff.train()
#     ckpt_path = CKPT_DIR + "/shallownet_v3_20.ckpt"
#     rff = torch.load(ckpt_path).to(device)
#     rff.test()

def find_latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
        return 0

    ep = 30
    while(not os.path.exists(CKPT_DIR + "/shallownet_v3" + f"_{ep}.ckpt")):
        ep -= 1
        if (ep <= 0):
            break
    return ep

if __name__ == "__main__":
    latest_checkpoint = 20#find_latest_checkpoint()

    if (latest_checkpoint == 0):
        rff = RFF().to(device)
        rff.train()
        rff.test()
    else:
        ckpt_path = CKPT_DIR + "/shallownet_v3" + f"_{latest_checkpoint}.ckpt"
        rff = torch.load(ckpt_path, map_location='cpu').to(device)
        #if (latest_checkpoint < num_epochs):
            #rff.train(start_epoch=latest_checkpoint)
        #rff.test()
        rff.test40()

