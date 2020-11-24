import tensorflow as tf
import os


saved_model_dir = "/Users/Daniele/Dropbox/MAGISTRALE/Second Year/Machine Learning for IOT/labs/IOT_Labs/lab3/ex3/out/models/" #abs path to model dir
tflite_model_dir = "/Users/Daniele/Dropbox/MAGISTRALE/Second Year/Machine Learning for IOT/labs/IOT_Labs/ex5/out/models_lite" # abs path to new model dir
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# part 0 - convert models to their lite counterparts

with open(tflite_model_dir, 'wb') as fp:
     fp.write(tflite_model)

     # Measure the tflite file size for each model (use getsize from os.path)
     #path = "some path"
     #size = os.path.getsize(path)
     #print(size)
     # copy the models to the board (already on the board)


# part 1

#Write a Python script that collect six consecutive values (sampling frequency = 1s) of
#temperature and humidity values with the DHT-11 sensor and make predictions with
#the tflite models.

# 0 - collect six consecutive values of temperature and humidity 

# 1 - put those in a pandas df

# 3 - load models

# 4 - run models with the test data

# 5 - evaluate forecasting accuracy

# Compare measured vs. predicted values. Example output:
# Measured: 20.0, 80.0 --- Predicted: 20.4,79.8 --- MAE: 0.4,0.2

# Write a Python script to measure the Latency (batch size=1) in ms for each model.
# Compute the average on 100 runs.

# part 2

