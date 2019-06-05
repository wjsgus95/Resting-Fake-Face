from torchvision import datasets
import torchvision
import torch
import torch.utils.data

class TestDataSet(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(TestDataSet, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

"""
# EXAMPLE USAGE:
# instantiate the dataset and dataloader
data_dir = "../uniform_trainset"
dataset = TestDataSet(data_dir, transform=torchvision.transforms.ToTensor()) # our custom dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False)

# iterate over data
for inputs, labels, paths in dataloader:
    # use the above variables freely
    #print(inputs, labels, paths)
    print(list(zip(labels, paths)))
"""
