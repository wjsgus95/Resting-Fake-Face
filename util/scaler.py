import os
import cv2

# Only for square images
class Scalar():
    def __init__(self):
        pass

    def get_img_of_size(self, img, size):
        return cv2.pyrDown(img, size)

    def convert_to_size(self, folder, newfolder, size):
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)  
        
        #count = 1 #debugging
        for filename in os.listdir(folder):
            # count += 1
            # if (count > 10):
            #     break  #debugging

            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                new_img = cv2.resize(img, dsize=(size, size))
                cv2.imwrite(newfolder + "/" + filename, new_img)

if __name__ == "__main__":
    scaler = Scalar()

    size = 128
    scaler.convert_to_size('trainset/fake', 'trainset/fake_new_images', size)
    scaler.convert_to_size('trainset/gan', 'trainset/gan_new_images', size)
    scaler.convert_to_size('trainset/real', 'trainset/real_new_images', size)

    pass