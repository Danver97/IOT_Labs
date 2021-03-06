import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
import wave
import time
import sys
import os 
import io

from scipy.io import wavfile
from scipy import signal


form_1 = pyaudio.paInt16 # resolution
samp_rate = 48000 # sampling rate
record_secs = 1 # seconds to record
dev_index = 0 # device index found by p.get_device_info_by_index(ii)
chunk = 4800  # 2^12 samples for buffer
chans = 1 # 1 channel
samples = 10
sample_ratio = 3
new_sample_rate = 16000
num_mel_bins = 40
lower_freq_mel = 20
upper_freq_mel = 4000
num_coefficients = 10



def pad(audio):
    zero_padding = tf.zeros([new_sample_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([new_sample_rate])

    return audio

def get_spectrogram(audio, frame_length = 256, frame_step = 128):
    stft = tf.signal.stft(audio, frame_length=frame_length,
            frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    return spectrogram

def get_mfccs(spectrogram):
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, 321, new_sample_rate,
        lower_freq_mel, upper_freq_mel)
    mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]

    return mfccs

def preprocess_with_stft(audio):
    audio = pad(audio)
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, [32, 32])

    return spectrogram

def preprocess_with_mfcc(audio):
    audio = pad(audio)
    spectrogram = get_spectrogram(audio,640,320)
    mfccs = get_mfccs(spectrogram)
    mfccs = tf.expand_dims(mfccs, -1)

    return mfccs

buffer = io.BytesIO()
audio_stream = pyaudio.PyAudio()
stream = audio_stream.open(format = form_1,
                    rate = samp_rate,
                    input_device_index = dev_index,
                    input = True,
                    channels = chans,
                    start = False)

print('recording...')
time.sleep(0.5)
stream.start_stream()
for ii in range(int((samp_rate/chunk)*record_secs)):
    buffer.write(stream.read(chunk))  
stream.stop_stream()
print('STOP') 
buffer.seek(0)
audio = np.frombuffer(buffer.getvalue(), dtype=np.int16)

tf_audio = signal.resample_poly(audio, 1, sample_ratio)
tf_audio = tf.convert_to_tensor(tf_audio, dtype=tf.float32)


stream.close()
audio_stream.terminate()

t = time.time()
audio_stft = preprocess_with_stft(tf_audio)
print("stft_preprocessing:",time.time() - t)
t = time.time()
audio_mfcc = preprocess_with_mfcc(tf_audio)
print("mfcc_preprocessing:",time.time() - t)

audio_stft = tf.expand_dims(audio_stft, axis=0)
audio_mfcc = tf.expand_dims(audio_mfcc, axis=0)
labels_silence = ['down','go','left','no','right','silence','stop','up','yes']   
labels = ['down','go','left','no','right','stop','up','yes']
for model_name in os.listdir('tflite_models'):
    print("===============================",model_name,"===============================")
    interpreter = tflite.Interpreter(model_path='tflite_models/'+model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if "stft" in model_name:
        interpreter.set_tensor(input_details[0]['index'], audio_stft)
    else:
        interpreter.set_tensor(input_details[0]['index'], audio_mfcc)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    if "silence" in model_name:
        prediction = labels_silence[np.argmax(prediction)]
    else:
        prediction = labels[np.argmax(prediction)] 
    print("Prediction:",prediction)