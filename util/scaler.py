import os
import cv2
def downsample_resize(img, size):
        img_size = img.shape[0]

        scale = int(img_size / size)
        while scale > 1:
            new_size = int(img.shape[0] / 2)
            img = cv2.pyrDown(img, dstsize=(new_size, new_size))
            scale = int(scale / 2)

        return cv2.resize(img, dsize=(size, size))

# Only for square images
class Scalar():
    def __init__(self):
        pass

    def convert_to_size(self, folder, newfolder, size):
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)  
        
        #count = 1 #debugging
        for filename in os.listdir(folder):
            #count += 1
            #if (count > 10):
                #break  #debugging

            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                #downsampling and resizing produces smoother images so doing that instead
                new_img = downsample_resize(img, size) #cv2.resize(img, dsize=(size, size))
                cv2.imwrite(newfolder + "/" + filename, new_img)

if __name__ == "__main__":
    scaler = Scalar()

    size = 128
    scaler.convert_to_size('trainset/fake', 'trainset/fake_new_images_ds', size)
    scaler.convert_to_size('trainset/gan', 'trainset/gan_new_images_ds', size)
    scaler.convert_to_size('trainset/real', 'trainset/real_new_images_ds', size)

    pass