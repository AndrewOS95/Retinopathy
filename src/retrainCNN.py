import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
#from keras.utils import multi_gpu_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, save_model, load_model

def split_data(X, y, train_test_size):
    """
    Split data into test and training datasets.

    INPUT
        X: NumPy array of arrays
        y: Pandas series, which are the labels for input array X
        train_test_size: size of test/train split. Value from 0 to 1

    OUPUT
        Four arrays: X_train, X_test, y_train, and y_test
    """
    return train_test_split(X, y, test_size=train_test_size, random_state=42)


def format_data(array, image_rows, image_cols, channels):
    """
    Reshapes the data into format for CNN.

    INPUT
        array: array of numpy arrays.
        image_rows: Image height
        image_cols: Image width
        channels: Specify if the image is grayscale (1) or RGB (3)

    OUTPUT
        Reshaped array of NumPy arrays.
    """
    return array.reshape(array.shape[0], image_rows, image_cols, channels)


def cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes):
    """
       Define and run the Convolutional Neural Network

       INPUT
           X_train array of numpy arrays
           X_test: array of numpy arrays
           y_train: label array
           y_test: label array
           kernel_size: kernel size
           nb_filters: filter size
           channels: specify if the image is grayscale (1) or RGB (3)
           nb_epoch: number of epochs
           batch_size: How many images to load into memory for training
           nb_classes: number of classes for classification

       OUTPUT
           Fitted CNN model
       """


    model = Sequential()

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(image_rows, image_cols, channels), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    #stop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=4, verbose=0, mode='auto')


    tensor_board = TensorBoard(log_dir='C:/work/Stage5/model', histogram_freq=0, write_graph=True, write_images=True)
    #Save checkpoint (Basically saves the weights)
    filepath = "C:/work/Stage5/model/weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, tensor_board]
    #Load previously trained model for retraining. If there is no model, then comment out.
    model = load_model('C:/work/model/DR_Two_Classes1.0.h5')


    #Fitting model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1,
              validation_split=0.2,
              class_weight='auto',
              #callbacks=[stop, tensor_board, callbacks_list]
              callbacks=callbacks_list)

    return model


def save_model(model, score, name):
    """
    Saves Keras model to an h5 file, based on precision_score

    INPUT
        model: Keras model object to be saved
        score: Score to determine if model should be saved.
        model_name: name of model to be saved
    """

    if score >= 0.7:
        print("Saving Model")
        model.save("C:/work/Stage5/model/" + name + str(round(score)) + ".h5")
    else:
        print("Model Not Saved.  Score: ", score)

if __name__ == '__main__':
    # Specify parameters before model is run.
    batch_size = 25
    nb_classes = 2
    nb_epoch = 5

    image_rows, image_cols = 256, 256
    channels = 3
    nb_filters = 32
    kernel_size = (8, 8)

    # Import data

    labels = pd.read_csv("C:/work/Stage5/labels/trainLabels_master_256_v2.csv")
    X = np.load("C:/work/Stage5/labels/X_train.npy")
    y = np.array([1 if l >= 1 else 0 for l in labels['level']])

    print("Splitting data into test/train datasets")
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    print("Formatting Data")
    X_train = format_data(X_train, image_rows, image_cols, channels)
    X_test = format_data(X_test, image_rows, image_cols, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    input_shape = (image_rows, image_cols, channels)

    print("Normalizing Data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    print("Training Model")

    model = cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size,
                     nb_classes)


    print("Predicting")
    y_pred = model.predict(X_test)

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)

    save_model(model=model, score=recall, model_name="DR_Two_Classes")
    print("Completed")