import pandas as pd
import numpy as np
import os
from PIL import Image, ImageFile
from skimage import io
from skimage.transform import resize, rotate
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_dir(dir):
    '''
    Creates a new folder in the specified directory if it doesn't exist,
    INPUT
        dir: Folder to be created
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

def crop_resize_imgs(path, new_path, cropX, cropY, image_size=256):
    '''
    Crops, resizes and stores all images from a selected directory into the new one
    INPUT
        path: Path of the current images.
        new_path: Where to store the new images.
        image_size: Size of new images.
    OUTPUT
        Images are cropped, resized and saved to the new directory.
    '''

    make_dir(new_path)
    directories = [l for l in os.listdir(path) if l != '.DS_Store']
    number = 0


    for item in directories:

        image = io.imread(path + item)

        Y,X,channel = image.shape

        makeX = X//2-(cropX//2)
        makeY = Y//2-(cropY//2)

        image = image[makeY:makeY+cropY, makeX:makeX+cropX]

        image = resize(image, (256,256))

        io.imsave(str(new_path + item), image)

        number += 1

        print("Saving: ", item, number)


def get_dark_images(new_path, dataframe):
    """
    Find black images
    INPUT
        new_path: path to the images that will be analyzed.
        dataframe: Pandas DataFrame that includes labeled image names.
        column: column in DataFrame query is evaluated against.
    OUTPUT
        Column indicating if the photo is black or not.
    """

    image_list = [i for i in dataframe['image']]
    return [1 if np.mean(np.array(Image.open(new_path + image))) == 0 else 0 for image in image_list]


if __name__ == '__main__':

    start_time = time.time()

    crop_resize_imgs(path='C:/work/dEYENET/train/train/',
                     new_path='C:/work/dEYENET/train_resized_256/',
                     cropxX=1800, cropY=1800, image_size=256)
    crop_resize_imgs(path='C:/work/dEYENET/test/test/',
                     new_path='C:/work/dEYENET/test_resized_256/',
                     cropX=1800, cropY=1800, image_size=256)

    train_label = pd.read_csv('C:/work/dEYENET/Stage2/labels/trainLabels.csv')
    train_label['image'] = [i + '.jpeg' for i in train_label['image']]
    train_label['black'] = np.nan


    train_label['black'] = get_dark_images('C:/work/dEYENET/Stage2/train_resized_256/', train_label)

    train_labels = train_label.loc[train_label['black'] == 0]

    train_label.to_csv('C:/work/dEYENET/Stage2/labels/trainLabels_master.csv', index=False, header=True)

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))