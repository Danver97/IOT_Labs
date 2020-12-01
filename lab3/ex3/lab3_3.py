import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--labels', type=int, required=True, help='model output')
args = parser.parse_args()
model_type = args.model
labels = args.labels

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

csv_path = "./data/jena_climate_2009_2016.csv"
df = pd.read_csv(csv_path)

column_indices = [2,5] # Temperature
# column_indices = [5,2] # Humidity
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.8)]
test_data = data[int(n*0.8):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6
LABEL_OPTIONS = labels


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
            labels = features[:, -1, self.label_options] # temperature, use 0 temp use 1 humidity
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -1, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, num_labels])

        inputs.set_shape([None, self.input_width, 2])

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

# 3.2
generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

if LABEL_OPTIONS == 2:
    units = 2
else:
    units = 1
# 3.3, ref slides 09 (sequential)
# A multilayer perceptron (MLP)
def modelMLP():
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(units = units)
        ])
    return model

# A convolutional neural network (CNN-1D)
def modelCNN1D():
    model = keras.Sequential([
        keras.layers.Conv1D(filters  = 64, kernel_size= 3, activation= 'relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(units = units)
        ])
    return model

# An LSTM
def modelLSTM():
    model = keras.Sequential([
        keras.layers.LSTM(units = 64),
        keras.layers.Flatten(),
        keras.layers.Dense(units = units)
        ])
    return model

# 3.7
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

# 3.4 train each model and evaluate the prediction error (see details on the hw)
utilityPlot = []
if model_type == 'mlp':
    model = modelMLP()
elif model_type == 'cnn':
    model = modelCNN1D()
elif model_type == 'lstm':
    model = modelLSTM()
    

print("====== Model considered: " + model_type + " =========")
if units == 1:
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )
elif units == 2:
    model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.MeanSquaredError(),
    metrics=MultiOutputMAE()
    )
    

history = model.fit(train_ds, validation_data= val_ds, epochs=20)
loss, mae = model.evaluate(test_ds)
print(mae)
print("MAE on the test set: " + str(mae))
tot_params = sum(len(v) for v in tf.compat.v1.trainable_variables())
utilityPlot.append((mae, tot_params, 'mae'))

# 3.5
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
model.save("./lab3/ex3/out/models/", signatures=concrete_func)


print("#Params",model.summary())
print(" === END MODEL ===")


# plot #params v mae for each model
if units == 1:
    dfPlot = pd.DataFrame(utilityPlot, columns =['mae', 'params', 'name']) 
    ax = sns.pointplot(x=dfPlot["mae"] , y=dfPlot["params"] , hue="name",data=dfPlot)

# 3.6
#Modify the script to train models for humidity forecasting. Repeat the experiments and the
#assessment. Comment the results.

# see above


# 3.8
# theoretical answer