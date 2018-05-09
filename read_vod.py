import cv2
import numpy as np
import os
from random import shuffle

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

PATH_TO_DATA = 'data/'

label_dic = {'soldier': 0, 'genji': 1, 'reaper': 2, 'ana': 3, 'bastion': 4, 'brigitte': 5, 'doomfist': 6, 'dva': 7, 'hanzo':8,
'junkrat': 9, 'lucio': 10, 'mccree': 11, 'mei': 12, 'mercy': 13, 'moira': 14, 'orisa': 15, 'pharah': 16, 'reinhardt': 17,
'roadhog': 18, 'sombra': 19, 'symmetra': 20, 'torbjorn': 21, 'tracer': 22, 'widowmaker': 23, 'winston': 24, 'zarya': 25,
'zenyatta': 26, 'babydva': 27}

def get_dataset(data_root_path, get_test_set=False):
    X = []
    y = []
    for folder in os.listdir(data_root_path):
        path_to_images = data_root_path + folder
        label = folder.split("_")[0]

        if not os.path.isdir(path_to_images):
            continue
        # skip when "test" is in folder and we don't want the test set
        if "test" in folder and not get_test_set:
            continue
        # skip when "test" in not in folder and we want the test set.
        if "test" not in folder and get_test_set:
            continue

        for image_name in os.listdir(path_to_images):
            # sometimes osx has some hidden files that fucks stuff up. ignore them!
            if ".jpg" not in image_name:
                continue
            X.append(path_to_images + "/" + image_name)
            y.append(label_dic[label])

    y = keras.utils.to_categorical(y, len(label_dic))

    zipped = list(zip(X, y))
    shuffle(zipped)
    X, y = zip(*zipped)
    # for q,w in zip(X,y):
    #     print(q, w)
    # print(len(X), len(y))
    return X, y

def load_images_for_model(X_batch, resize_to_720P=True):
    # X_batch is just a bunch of file names. We need to load the image to pass it to a net!
    X_loaded = []

    for path in X_batch:
        img = cv2.imread(path)

        # pure 1080p resized to 720p.
        if resize_to_720P:
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
            # crop the gun, in the bottom right.
            img = img[612:668, 1090:1210]

        # pure 1080p video.
        else:
            # crop the gun, in the bottom right.
            img = img[920:1000, 1650:1820]

        # resize before passing to net. i keep the aspect ratio 17:8
        img = cv2.resize(img, (85, 40))
        # dividing by 255 leads to faster convergence through normalization.
        X_loaded.append(np.array(img)/(255))

    cv2.imwrite('t1.jpg', X_loaded[0] * 255)
    return X_loaded

def get_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(40, 85, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(27))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()
    return model

# i could have used a batch generator here, but wanted to try this way, without one.
# a batch generator is wayyy better because it loads data in parallel and caches it.
# but this fun :)
def train():
    X, y = get_dataset(PATH_TO_DATA)

    # X_train = X[0:20]
    # y_train = y[0:20]
    X_train = X[0:int(len(X) * 0.8)]
    y_train = y[0:int(len(X) * 0.8)]

    X_val = X[int(len(X) * 0.8):]
    y_val = y[int(len(X) * 0.8):]

    model = get_model()
    num_epochs = 10
    batch_size = 8

    print("Beginning training!")
    # print("Training set is size %d and Val set is size %d" % (len(X_train), len(X_val)))

    for e in range(num_epochs):
        print("On epoch %d" % e)

        losses = []
        accs = []
        for step_num in range(0, len(X_train), batch_size):
            X_batch = np.asarray(load_images_for_model(X_train[step_num: step_num + batch_size]))
            y_batch = np.asarray(y_train[step_num: step_num + batch_size])
            history = model.fit(X_batch, y_batch, batch_size=batch_size, epochs=1, verbose=0)
            losses.append(history.history['loss'][0])
            accs.append(history.history['acc'][0])

        print(sum(losses)/len(losses), sum(accs)/len(accs))

        # now do the same stuff again, but for validation set!
        losses = []
        accs = []
        for step_num in range(0, len(X_val), batch_size):
            X_batch = np.asarray(load_images_for_model(X_val[step_num: step_num + batch_size]))
            y_batch = np.asarray(y_val[step_num: step_num + batch_size])
            loss = model.evaluate(X_batch, y_batch, batch_size=batch_size, verbose=0)
            losses.append(loss[0])
            accs.append(loss[1])

        print(sum(losses)/len(losses), sum(accs)/len(accs))
        # save model at the end of every epoch.
        model.save('saved_model.h5')


if __name__ == '__main__':
    train()
