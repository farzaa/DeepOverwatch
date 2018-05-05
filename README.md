# DeepOverwatch

This repo readme only discusses how to run the code. To find out what DeepOverwatch actually is, please read the blog post [here](https://medium.com/@farzatv/deepoverwatch-combining-tensorflow-js-overwatch-and-music-1a84d4598bc0) :). 

![lol](https://s3-us-west-2.amazonaws.com/mood1995/medium/Screen+Shot+2018-05-04+at+8.23.31+PM.png)

If you have **questions** at any point, DM me on Twitter [@farzatv](https://twitter.com/FarzaTV).

DeepOverwatch was already trained by me on the 27 heroes that were released as of May 4th, 2018. You can get the pretrained Keras model [here](https://s3-us-west-2.amazonaws.com/mood1995/all_heroes_model.h5). To load this model in Keras do:

```
from keras.models import load_model
model = load_model('MODEL_NAME.h5')
```

Than, to predict do:
```
model.predict(INSERT_LOADED_IMAGE_HERE)
```

The only preprocessing I do can be found in the ```load_images_for_model``` method. Be sure to do this same preprocessing before trying to ```predict``` on an image. DeepOverwatch only works on screenshots that were taken in a game running at 1920x1080. You *can not* run the game at a different resolution and resize the image to 1920x1080.


If you want the dataset and want to train your own neural net you can get it here. Its a fat 10GB file. Once you get it, extract the the mp4 files within the ZIP file into a folder name ```clips``` and place this folder in the root directory of this project. Than, simply uncomment the ```convert_clips()``` line and go! This will take a while. 


*gl hf*

**-Farza**





