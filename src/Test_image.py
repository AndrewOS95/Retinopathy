from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import numpy as np
import cv2
import pandas as pd
from keras.models import Sequential, save_model, load_model
from PIL import Image
import csv
import os

def change_image_name(df, column):

    return [i + '.jpeg' for i in df[column]]

def convert_images_to_arrays(file_path, df):
    lst_imgs = [l for l in df['image']]
    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs])

def save_to_array(arr_name, arr_object):
    return np.save(arr_name, arr_object)


def reshape_data(arr, img_rows, img_cols, channels):
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


if __name__ == "__main__":

    labels = pd.read_csv("C:/work/labels/mylabel.csv")
    print("Writing Train Array")
    X_test= convert_images_to_arrays('C:/work/train_30/', labels)
    image_name = os.path.basename('C:/work/train_30/30_left')
    username = str(image_name)
    print(username)
    print(X_test.shape)
    print("Saving Train Array")
    save_to_array('C:/work/train_30/X_test.npy', X_test)

    model = load_model('C:/work/Stage1/model1/DR_Two_Classes1.0.h5')
    X_test = np.load("C:/work/train_30/X_test.npy")
    Y = np.array([1 if l >= 1 else 0 for l in labels['level']])
    with open('C:/work/labels/trainLabels.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            name = str(row[0])
            level = row[1]
            if name == username:
                name = username
                Y = float(level)
                if Y >= 1:
                    Y = 1
                else:
                    Y = 0
                break

    img_rows = 256
    img_cols = 256
    channels = 3

    print("Reshaping Data")
    X_test = reshape_data(X_test, img_rows, img_cols, channels)
    print("X_test Shape: ", X_test.shape)

    print("Normalizing Data")
    X_test = X_test.astype('float32')
    print('retinopathy greade ',Y)
    prediction = model.predict(X_test)
    print("Analyzing picture ", name)
    print(prediction)
    if Y ==1 and prediction.flat[0] > 0.5 :
        print("Diseased eye, Confidence of:", prediction.flat[0])
    if Y ==1 and prediction.flat[0] < 0.5 :
        print ("Normal eye, Confidence of:", prediction.flat[1])
    if Y ==0 and prediction.flat[0] > 0.5 :
        print("Diseased eye, Confidence of:", prediction.flat[0])
    if Y ==0 and prediction.flat[0] < 0.5 :
        print("Normal eye, Confidence of:", prediction.flat[1])