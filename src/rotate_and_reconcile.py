import pandas as pd
from skimage import io
from skimage.transform import rotate
import cv2
import time
import os

def rotate(new_path, degrees, image_list):
    """
    Rotates images. Degrees of rotation are specified

    INPUT:
        new_path represents the path to the folder containing resied images
        degrees represent the degrees of rotation
        image_list is the list of image strings
    OUTPUT:
        Roated images by the degrees specified
    """
    for i in image_list:
        image = io.imread(new_path + str(i) + '.jpeg')
        image = rotate(image, degrees)
        io.imsave(new_path, str(i) + '_' + str(degrees) + '.jpeg', image)


def mirror_img(new_path, direction, image_list):
    """
    Mirrors images

    INPUT:
        new_path represents the path to the folder containing resized images
        direction represents the mirroring direction
        image_list is the list of image strings
    OUTPUT:
        mirrored images

    """
    for i in image_list:
        image = cv2.imread(new_path + str(i) + '.jpeg')
        image = cv2.flip(image, 1)
        cv2.imwrite(new_path + str(i) + '_mir' + '.jpeg', image)


def get_image_list(new_path):
    """
    Reads in filed from the new path into a list
    INPUT:
        new_path is the filed path containing the resized and rotated images
    OUTPUT:
        is the list of image strings
    """

    return [i for i in os.listdir(new_path) if i!='.DS_Store']


if __name__ == '__main':

    start_time = time.time()

    train_labels = pd.read_csv("C:/work/dEYENET/labels/trainLabels_master.csv")
    train_labels['image'] = train_labels['image'].str.rstrip('.jpeg')
    train_labels_no_DR = train_labels[train_labels['level'] == 0]
    train_labels_DR = train_labels[train_labels['level'] >= 1]

    image_list_no_DR = [i for i in train_labels_no_DR['image']]
    image_list_DR = [i for i in train_labels_DR['image']]


    print("Mirroring Non-DR Images")
    mirror_img('C:/work/dEYENET/train_resized_256/', 1, image_list_no_DR)

    # Rotate all images that have any level of DR
    print("Rotating 90 Degrees")
    rotate('C:/work/dEYENET/train_resized_256/', 90, image_list_DR)

    print("Rotating 120 Degrees")
    rotate('C:/work/dEYENET/train_resized_256/', 120, image_list_DR)

    print("Rotating 180 Degrees")
    rotate('C:/work/dEYENET/train_resized_256/', 180, image_list_DR)

    print("Rotating 270 Degrees")
    rotate('C:/work/dEYENET/train_resized_256/', 270, image_list_DR)

    print("Mirroring DR Images")
    mirror_img('C:/work/dEYENET/train_resized_256/', 0, image_list_DR)

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))

    train_label = pd.read_csv("C:/work/dEYENET/Stage3/labels/trainLabels_master.csv")
    image_list = get_image_list('C:/work/dEYENET/Stage3/train_resized_256/')

    new_train_label = pd.DataFrame({'image': image_list})
    new_train_label['image2'] = new_train_label.image

    # Remove the suffix from the image names.
    new_train_label['image2'] = new_train_label.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

    # Strip and add .jpeg back into file name
    new_train_label['image2'] = new_train_label.loc[:, 'image2'].apply(
        lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')

    new_train_label.columns = ['train_image_name', 'image']
    train_labels = pd.merge(train_labels, new_train_label, how='outer', on='image')
    train_labels.drop(['black'], axis=1, inplace=True)
    train_labels = train_labels.dropna()
    print(train_labels.shape)
    print("Writing CSV")
    train_labels.to_csv('C:/work/dEYENET/Stage3/labels/trainLabels_master_256_v2.csv', index=False, header=True)
