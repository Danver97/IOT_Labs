import tensorflow as tf
from tensorflow import keras
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, required=True, help="model name")
parser.add_argument('--mfcc', action='store_true', help="use MFCCs")
parser.add_argument('--silence', action='store_true', help="add silence")
args = parser.parse_args()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

if args.silence is True:
    data_dir = os.path.join('.', 'data', 'mini_speech_commands_silence')
else:
    zip_path = keras.utils.get_file(
        origin = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname = "mini_speech_commands.zip",
        extract = True,
        cache_dir = ".",
        cache_subdir = "data"
    )
    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
numsamples = len(filenames)

if args.silence is True:
    total = 9000
else:
    total = 8000

train_files = filenames[:int(total*0.8)]
val_files = filenames[int(total*0.8):int(total*0.9)]
test_files = filenames[int(total*0.9):]

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != "README.md"]

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
    num_mel_bins=None, lower_frequency=None, upper_frequency=None, num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1
        
        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency
            )
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft


    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype = tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram
    
    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds

STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
                'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40, 
                'num_coefficients': 10}

if args.mfcc is True:
    options = MFCC_OPTIONS
    strides = [2, 1]
else:
    options = STFT_OPTIONS
    strides = [2, 2]

generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

if args.silence is True:
    units = 9
else: 
    units = 8

mlp = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 256, activation = "relu"),
    tf.keras.layers.Dense(units = 256, activation = "relu"),
    tf.keras.layers.Dense(units = 256, activation = "relu"),
    tf.keras.layers.Dense(units = units)
])

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 128, kernel_size = [3, 3], strides = strides, use_bias = False),
    tf.keras.layers.BatchNormalization(momentum = 0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters = 128, kernel_size = [3, 3], strides = [1, 1], use_bias = False),
    tf.keras.layers.BatchNormalization(momentum = 0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters = 128, kernel_size = [3, 3], strides = [1, 1], use_bias = False),
    tf.keras.layers.BatchNormalization(momentum = 0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units = units)
])

dscnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 256, kernel_size = [3, 3], strides = strides, use_bias = False),
    tf.keras.layers.BatchNormalization(momentum = 0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size = [3, 3], strides = [1, 1], use_bias = False),
    tf.keras.layers.Conv2D(filters = 256, kernel_size = [1, 1], strides = [1, 1], use_bias = False),
    tf.keras.layers.BatchNormalization(momentum = 0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size = [3, 3], strides = [1, 1], use_bias = False),
    tf.keras.layers.Conv2D(filters = 256, kernel_size = [1, 1], strides = [1, 1], use_bias = False),
    tf.keras.layers.BatchNormalization(momentum = 0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units = units)
])


MODELS = { 'mlp': mlp, 'cnn': cnn, 'dscnn': dscnn }
model = MODELS[args.model]

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
metrics = [tf.metrics.SparseCategoricalAccuracy()]

checkpoint_filepath = './checkpoints/kws_{}_{}/weights'.format(args.model, args.mfcc)
cp = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    save_best_only=True
)

if not os.path.exists(os.path.dirname(checkpoint_filepath)):
    os.makedirs(os.path.dirname(checkpoint_filepath))

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[cp])

print(model.summary())

model.load_weights(checkpoint_filepath)
loss, accuracy = model.evaluate(test_ds)
print('Accuracy: {:.2f}%'.format(accuracy*100))

saved_model_dir = './models/kws_{}_{}'.format(args.model, args.mfcc)
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
model.save(saved_model_dir)