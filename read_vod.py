import cv2
import numpy as np
import os

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.callbacks import TensorBoard

from sklearn.preprocessing import LabelEncoder

from random import shuffle


PATH_TO_CLIPS = 'clips/'
PATH_TO_DATA = 'data/'

label_dic = {'soldier': 0, 'genji': 1, 'reaper': 2}

from keras.utils import to_categorical


def get_train_dataset():
    X = []
    y = []
    for folder in os.listdir(PATH_TO_DATA):
        path_to_images = PATH_TO_DATA + folder
        label = folder.split("_")[0]

        if not os.path.isdir(path_to_images):
            continue
        if "test" in folder:
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

def load_images_for_model(X_batch):
    # X_batch is just a bunch of file names. We need to load the image to pass it to a net!
    X_loaded = []

    for path in X_batch:
        img = cv2.imread(path)
        # crop the gun, in the bottom right.
        img = img[920:1000, 1650:1820]
        img = cv2.resize(img, (85, 40))

        X_loaded.append(img)

    cv2.imwrite('t1.jpg', X_loaded[0])
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

    model.add(Dense(3))
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
    X, y = get_train_dataset()

    X_train = X[0:int(len(X) * 0.8)]
    y_train = y[0:int(len(X) * 0.8)]

    X_val = X[int(len(X) * 0.8):]
    y_val = y[int(len(X) * 0.8):]


    model = get_model()
    num_epochs = 25
    batch_size = 8


    print("Beginning training!")
    print("Training set is size %d and Val set is size %d" % (len(X_train), len(X_val)))

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
























def parse_video(file_name, full_path, original_count):
    count = original_count

    video = cv2.VideoCapture(full_path)
    success, frame = video.read()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading %s, %d seconds long with FPS %d and total frame count %d ' % (file_name, total_frame_count/fps, fps, total_frame_count))

    while success:
        count += 1
        # original video at 30FPS, halve FPS to reduce redundancy in training data.
        success, frame = video.read()
        if not success:
            break

        if not count % 3 == 0:
            continue

        if count % 300 == 0:
            print('Currently at frame ', count-original_count)

        # finish up by saving the image to either our train set or our test set.
        # wheres the validation set? i just take a split of my train set. no need to complicate things here.
        if "test" in file_name:
            cv2.imwrite("data/" + label + "_test" + '/' + str(original_count + count) + '.jpg', frame)
        else:
            cv2.imwrite("data/" + label + "_train" + '/' + str(original_count + count) + '.jpg', frame)

    video.release()

def convert_clips():
    for clip_name in os.listdir(PATH_TO_CLIPS):
        # the name of the video holds the label, ex soldier_1.mp4, genji_7.mp4, etc.
        label = clip_name.split("_")[0]

        if not os.path.isdir("data/" + label + "_train"):
            os.mkdir("data/" + label + "_train")

        if not os.path.isdir("data/" + label + "_test"):
            os.mkdir("data/" + label + "_test")

        if "test" in clip_name:
            parse_video(clip_name, PATH_TO_CLIPS + clip_name, len(os.listdir("data/" + label + "_test")))
        else:
            parse_video(clip_name, PATH_TO_CLIPS + clip_name, len(os.listdir("data/" + label + "_train")))

train()
