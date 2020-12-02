import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--labels', type=int, required=True, help='model output')
args = parser.parse_args()


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6
LABEL_OPTIONS = args.labels


class WindowGenerator:
    def __init__(self, input_width, label_options, mean, std):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        input_indeces = np.arange(self.input_width)
        inputs = features[:, :-1, :]

        if self.label_options < 2:
            labels = features[:, -1, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -1, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width+1,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name,**kwargs)
        self.total = self.add_weight('total',initializer='zeros', shape=[2])
        self.count = self.add_weight('count',initializer='zeros')

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error,axis=0)
        self.total.assign_add(error)
        self.count.assign_add(1)
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result
        
generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

if LABEL_OPTIONS < 2:
    metric = tf.keras.metrics.MeanAbsoluteError()
    units = 1
else:
    metric = MultiOutputMAE()
    units = 2

filepath = args.model
if filepath == 'mlp':
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(6, 2)),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(units)
    ])
elif filepath == 'cnn':
    model = keras.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.Flatten(input_shape=(64,)),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units)
    ])
elif filepath == 'lstm':
    model = keras.Sequential([
        keras.layers.LSTM(64),
        keras.layers.Flatten(),
        keras.layers.Dense(units)
    ])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=metric
)
model.fit(train_ds, validation_data= val_ds, epochs=20)
model.summary()

if LABEL_OPTIONS == 0:
    filepath = filepath + "_temperature"
elif LABEL_OPTIONS == 1:
    filepath = filepath + "_humidity"
else:
    filepath = filepath + "_multiple"

if os.path.exists(filepath) is False:
    os.makedirs(filepath)
model.save(filepath)

# lab 4- @todo test
tf.data.experimental.save(train_ds, './th_train')
tf.data.experimental.save(val_ds, './th_val')
tf.data.experimental.save(test_ds, './th_test')

loss, acc = model.evaluate(test_ds)

if LABEL_OPTIONS<2:
    print(f"TEST: Loss {round(loss,3)}, Accuracy {round(acc,3)}")
    with open( filepath+"/TEST_result.log", "w" ) as f:
        f.write(f"Loss {round(loss,3)}, Accuracy {round(acc,3)}")
else:
    print(f"TEST: Loss {round(loss,3)}, Accuracy(T) {round(acc[0],3)}, Accuracy(H) {round(acc[1],3)}")
    with open( filepath+"/TEST_result.log", "w" ) as f:
        f.write(f"Loss {round(loss,3)}, Accuracy(T) {round(acc[0],3)}, Accuracy(H) {round(acc[1],3)}")