import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

# from common.constants import *
INPUT_SIZE = 128

TRAINSET_DIR = "../uniform_trainset"
FAKE_DIR = TRAINSET_DIR + "/Fake"
TRUE_DIR = TRAINSET_DIR + "/True"
GAN_DIR  = TRAINSET_DIR + "/gan"
TESTSET_DIR = "../dataset"


# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 128
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
class VGGNet(torch.nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        # 128 x 128 x 3
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, 
            kernel_size=3, stride=1, padding=1, bias=True) # 128 x 128 x 64
        self.conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, 
            kernel_size=3, stride=1, padding=1, bias=True) # 128 x 128 x 64
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 64 x 64 x 64
        self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, 
            kernel_size=3, stride=1, padding=1, bias=True) # 64 x 64 x 128
        self.conv2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, 
            kernel_size=3, stride=1, padding=1, bias=True) # 64 x 64 x 128 
        # self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) 32 x 32 x 128


        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, 
            kernel_size=3, stride=1, padding=1, bias=True) # 32 x 32 x 256
        self.conv3_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, 
            kernel_size=3, stride=1, padding=1, bias=True) # 32 x 32 x 256
        # self.conv3_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, 
        #     kernel_size=3, stride=1, padding=1, bias=True)
        # self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) 16 x 16 x 256

        self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, 
            kernel_size=3, stride=1, padding=1, bias=True) # 16 x 16 x 512
        self.conv4_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, 
            kernel_size=3, stride=1, padding=1, bias=True) # 16 x 16 x 512
        self.conv4_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, 
            kernel_size=3, stride=1, padding=1, bias=True) # 16 x 16 x 512
        # self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) 8 x 8 x 512

        self.conv5_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, 
            kernel_size=3, stride=1, padding=1, bias=True) # 8 x 8 x 512
        self.conv5_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, 
            kernel_size=3, stride=1, padding=1, bias=True) # 8 x 8 x 512
        self.conv5_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, 
            kernel_size=3, stride=1, padding=1, bias=True) # 8 x 8 x 512
        # self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 4 x 4 x 512


        self.fc1 = torch.nn.Linear(4*4*512, 4096)
        self.fc2 = torch.nn.Linear(4096, 1000)
        self.fc3 = torch.nn.Linear(1000, 3)
        self.softmax = torch.nn.Softmax(dim=1)
        # self.softmax = torch.nn.Softmax2d()

        #torch.save(self, ckpt_path)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max_pool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max_pool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        # self.conv3_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, 
        #     kernel_size=3, stride=1, padding=1, bias=True)
        x = self.max_pool(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.max_pool(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.max_pool(x)

        x = x.view(-1, 4*4*512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x

    def train(self):
        optimizer = optimize_getter(self, learning_rate)

        # Train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            print("epoch = %d" % epoch)
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
    vgg = VGGNet().to(device)
    print("training")
    vgg.train()
    vgg.test()

