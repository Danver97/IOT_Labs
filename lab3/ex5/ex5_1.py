import argparse
import time
import adafruit_dht 
import numpy as np
from board import D4
from time import sleep

import tensorflow.lite as tflite

parser = argparse.ArgumentParser()
parser.add_argument('--input_model', type=str, required=True, help='model name')
args = parser.parse_args()

frq = 1
samples = 6
train_set = [] 
dht_device = adafruit_dht.DHT11(D4)

print("working...")
for i in range(samples):
    temperature = dht_device.temperature
    humidity = dht_device.humidity 
    train_set.append([temperature,humidity])
    sleep(frq)

t, h = dht_device.temperature, dht_device.humidity
train_set = np.array(train_set, dtype=np.float32)
train_set = np.expand_dims(train_set, axis=0)
mean = [ 9.107597, 75.904076]
std = [ 8.654227, 16.557089]
train_set = (train_set - mean) / std
train_set = train_set.astype(np.float32)

print(train_set)

interpreter = tflite.Interpreter(model_path=args.input_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], train_set)
interpreter.invoke()
t0 = time.time()
prediction = interpreter.get_tensor(output_details[0]['index'])
t1 = time.time()

print(f"Prediction: {prediction} - True value: {t}, {h} - latency: {t1-t0:.2f} s")