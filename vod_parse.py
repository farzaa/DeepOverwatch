import cv2
import numpy as np
import os
from random import shuffle

PATH_TO_CLIPS = 'clips/'

def parse_video(file_name, full_path, original_count, label):
    count = original_count

    video = cv2.VideoCapture(full_path)
    success, frame = video.read()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading %s, %d seconds long with FPS %d and total frame count %d ' % (file_name, total_frame_count/fps, fps, total_frame_count))

    while success:
        count += 1
        # original video at 30FPS, FPS = (1/3) * FPS to reduce redundancy in training data.
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

# this methods runs one time at the very beginning.
# it takes the clips, converts them to images, and saves the images to train/test folders.
def convert_clips():
    for clip_name in os.listdir(PATH_TO_CLIPS):
        if ".mp4" not in clip_name:
            continue
        # the name of the video holds the label, ex soldier_1.mp4, genji_7.mp4, etc.
        label = clip_name.split("_")[0]

        if not os.path.isdir("data/" + label + "_train"):
            os.mkdir("data/" + label + "_train")

        if not os.path.isdir("data/" + label + "_test"):
            os.mkdir("data/" + label + "_test")

        # test files get placed in a different folder.
        if "test" in clip_name:
            parse_video(clip_name, PATH_TO_CLIPS + clip_name, len(os.listdir("data/" + label + "_test")), label)
        else:
            parse_video(clip_name, PATH_TO_CLIPS + clip_name, len(os.listdir("data/" + label + "_train")), label)

convert_clips()
