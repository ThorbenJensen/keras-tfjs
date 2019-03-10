"""Download Keras model, test it, and save to Tensorflow.JS format."""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications import mobilenet
from tensorflow.python.keras.preprocessing import image
from pprint import pprint
import tensorflowjs as tfjs

TEST_IMG = 'data/bike.jpg'
SAVE_PATH = "output/mobilenet"

# %% Load model

model = tf.keras.applications.MobileNet(weights='imagenet')

# %% Predict for image

def process_image(img_path):
  img_raw = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img_raw)
  img_array = np.expand_dims(img_array, axis=0)
  img_processed = mobilenet.preprocess_input(img_array)
  return img_processed

img = process_image(TEST_IMG)

pred = model.predict(img)
pred_decoded = mobilenet.decode_predictions(pred)
pprint(pred_decoded)

# %% Save model as as tf.js model

tfjs.converters.save_keras_model(model, SAVE_PATH)
