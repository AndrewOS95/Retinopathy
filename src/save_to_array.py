import time

import numpy as np
import pandas as pd
from PIL import Image


def change_image_name(dataframe, column):
    """
    Adds'.jpeg' suffix to image names in the DataFrame

    INPUT
        dataframe: Pandas DataFrame, including columns to be altered.
        column: The column that will be changed. Takes a string input.

    OUTPUT
        Pandas dataframe with the column changed for suffix inclusion.
    """
    return [i + '.jpeg' for i in dataframe[column]]


def convert_images_to_arrays_train(path, dataframe):
    """
    Converts all images to arrays then append each array to a new numpy array

    INPUT
        file_path: Specified path required for train images.
        dataframe: Pandas DataFrame being used to assist file imports.

    OUTPUT
        Array of numpy image arrays.
    """

    image_list = [l for l in dataframe['train_image_name']]

    return np.array([np.array(Image.open(path + image)) for image in image_list])


def save_to_array(array_name, array_object):
    """
    Saves data object as a NumPy file. Used for saving train and test arrays.

    INPUT
        array_name: The name of the file to save. Takes directory string.
        array_object: NumPy array of arrays saved as a NumPy file.

    OUTPUT
        Array of numpy image arrays.
    """
    return np.save(array_name, array_object)


if __name__ == '__main__':
    start_time = time.time()


    labels = pd.read_csv("C:/work/Stage5/labels/trainLabels_master_256_v2.csv")
    print("Writing Train Array")

    X_train = convert_images_to_arrays_train('C:/work/Stage5/train_resized_256/', labels)
    print(X_train.shape)

    print("Saving Train Array")

    save_to_array('C:/work/Stage5/labels/X_train.npy', X_train)
    print("--- %s seconds ---" % (time.time() - start_time))
