import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from common.constants import *

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
# Optimize function needs nn.Module instance on construction, hence callback function is used.
optimize_getter = lambda m, m_lr: optim.Adam(m.parameters(), lr=m_lr)

# Dataset
#train_dataset = torchvision.datasets.FashionMNIST("./", train=True, transform=transforms.ToTensor(), download=True)
train_dataset = torchvision.datasets.ImageFolder( root=TRAINSET_DIR, transform=torchvision.transforms.ToTensor())
#test_dataset = torchvision.datasets.FashionMNIST("./", train=False, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder( root=TESTSET_DIR, transform=torchvision.transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Resting Fake Face Model
class RFF(torch.nn.Module):
    def __init__(self):
        super(RFF, self).__init__()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()

        """
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        """

        # Block 1
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, bias=True)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)

        # Block 2
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, bias=True)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2, bias=True)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)

        # Block 3
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=2, bias=True)
        self.conv8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, bias=True)

        # Block 4
        self.fc1 = torch.nn.Linear(2048, 128)
        self.batchnorm = nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 2)

        #torch.save(self, ckpt_path)

    def forward(self, x):
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        # Flatten
        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        """
        # Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool1(x)
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
        x = self.maxpool2(x)
        x = self.dropout(x)

        # Block 3
        x = self.conv7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Block 4
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

    def train(self):
        optimizer = optimize_getter(self, learning_rate)

        # Train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images)
                # If L1 loss function is applied, do one hot encoding.
                if type(loss_func) == type(nn.L1Loss()):
                    labels = F.one_hot(labels, num_classes=10)
                    labels = labels.type(torch.FloatTensor).to(device)
                loss = loss_func(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

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

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == "__main__":

