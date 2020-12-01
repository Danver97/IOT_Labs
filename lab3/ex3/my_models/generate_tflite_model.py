import argparse
import os
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='model input')
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(args.input_dir)
tflite_model = converter.convert()
tflite_model_dir = 'tflite_'+args.input_dir

with open(args.input_dir+'.tflite', 'wb') as fp:
    fp.write(tflite_model)

size = os.path.getsize(tflite_model_dir)
print('size:', size)