import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from scaler import *

# adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return image

def crop_images(folder, newfolder, size):
    crop = RandomCrop(size)
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)  
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            for i in range(5): # perform multiple crops
                new_img = crop(img)
                cv2.imwrite(newfolder + "/" + filename + "_cropped%d.jpg" % i, 
                    new_img)


if __name__ == "__main__":
    # want to crop from smaller image
    # scaler = Scalar()
    # size = 180
    # scaler.convert_to_size('../trainset/fake', 
    #     '../trainset/fake_new_images_ds_180', size)
    # scaler.convert_to_size('../trainset/gan', 
    #     '../trainset/gan_new_images_ds_180', size)
    # scaler.convert_to_size('../trainset/real', 
    #     '../trainset/real_new_images_ds_180', size)

    size = 128
    crop_images('../trainset/fake_new_images_ds_180', "../trainset/fake_cropped", size)
    crop_images('../trainset/gan_new_images_ds_180', "../trainset/gan_cropped", size)
    crop_images('../trainset/real_new_images_ds_180', "../trainset/real_cropped", size)


