import cv2
import numpy as np
import os
from random import shuffle

import keras
from keras.models import load_model

from read_vod import get_dataset, load_images_for_model

PATH_TO_DATA = 'data/'

label_dic = {'soldier': 0, 'genji': 1, 'reaper': 2, 'ana': 3, 'bastion': 4, 'brigitte': 5, 'doomfist': 6, 'dva': 7, 'hanzo':8,
'junkrat': 9, 'lucio': 10, 'mccree': 11, 'mei': 12, 'mercy': 13, 'moira': 14, 'orisa': 15, 'pharah': 16, 'reinhardt': 17,
'roadhog': 18, 'sombra': 19, 'symmetra': 20, 'torbjorn': 21, 'tracer': 22, 'widowmaker': 23, 'winston': 24, 'zarya': 25,
'zenyatta': 26, 'babydva': 27}


# load model
model = load_model('trained_720_model.h5')

i = 0
batch_size = 32

losses = []
accs = []
X, y = get_dataset(PATH_TO_DATA, get_test_set=True)

for i in range(0, len(X), batch_size):
    X_batch = np.asarray(load_images_for_model(X[i: i + batch_size], resize_to_720P=True))
    y_batch = np.asarray(y[i: i + batch_size])

    loss = model.evaluate(X_batch, y_batch, batch_size=batch_size, verbose=0)
    losses.append(loss[0])
    accs.append(loss[1])

    print(sum(losses)/len(losses), sum(accs)/len(accs))
