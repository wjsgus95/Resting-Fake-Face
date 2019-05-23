import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

# Checkpoint path
ckpt_path = "./lenet5.ckpt"

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
train_dataset = torchvision.datasets.FashionMNIST("./", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.FashionMNIST("./", train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Fully Connected Network Model
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        pass

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def train(self):
        optimizer = optimize_getter(self, learning_rate)
        # Train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images)
                loss = loss_func(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    def test(self):
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# LeNet5 Model
class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.relu = nn.ReLU() 

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

        torch.save(self, ckpt_path)

    def forward(self, x):
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


def analyze_train():
    global batch_size, learning_rate, num_epochs, loss_func, optimize_getter
    global train_loader, test_loader

    # Hyperpameter candidates.
    batch_size_list = [25, 50, 100, 200]
    learning_rate_list = [0.0005, 0.001, 0.002, 0.004]
    loss_func_list = [nn.L1Loss(), nn.CrossEntropyLoss()]
    optimize_getter_list = [lambda m, m_lr: optim.Adam(m.parameters(), lr=m_lr),
                          lambda m, m_lr: optim.SGD(m.parameters(), lr=m_lr, momentum=0.9)]


    # Iterate over every possible comibination of hyperparamters.
    for m_batch_size in batch_size_list:
        batch_size = m_batch_size
        # Need different loader for different batch size
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for m_learning_rate in learning_rate_list:
            learning_rate = m_learning_rate
            for m_loss_func in loss_func_list:
                loss_func = m_loss_func
                for m_optimize_getter in optimize_getter_list:
                    optimize_getter = m_optimize_getter
                    lenet5 = torch.load(ckpt_path).to(device)

                    # Print hyperparmeter setting of this round.
                    print()
                    print("[Hyperparameters]")
                    print(f"batch_size = {batch_size}, learning_rate = {learning_rate}")
                    print(f"loss = {loss_func}")
                    print(f"optimizer = {optimize_getter(lenet5, learning_rate)}")

                    # Train and then infer.
                    lenet5.train()
                    lenet5.test()



if __name__ == "__main__":
    # Task 1
    #fc = FC().to(device)
    lenet5 = LeNet5().to(device)

    # Train and test FC network
    #print(); print("Training/Testing FCN...")
    #fc.train()
    #fc.test()

    # Train and test Lenet-5
    print(); print("Training/Testing LeNet-5...")
    lenet5.train()
    lenet5.test()
    # End of Task 1

    # Task 2
    #print(); print("Training/Testing LeNet-5 with hyperparameter tweaks...")
    #analyze_train()
    #TODO: print stats in csv format to plot them
    # End of Task 2
