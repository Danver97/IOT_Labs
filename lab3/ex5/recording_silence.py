import pyaudio
import numpy as np
import wave
import time
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

buffer = io.BytesIO()
audio_stream = pyaudio.PyAudio()
stream = audio_stream.open(format = form_1,
                    rate = samp_rate,
                    input_device_index = dev_index,
                    input = True,
                    channels = chans,
                    start = False)

for sample in range(1000):
    print(sample)
    stream.start_stream()
    for ii in range(int((samp_rate/chunk)*record_secs)):
        buffer.write(stream.read(chunk))  
    stream.stop_stream()  
    buffer.seek(0)
    audio = np.frombuffer(buffer.getvalue(), dtype=np.int16)
    tf_audio = signal.resample_poly(audio, 1, sample_ratio)
    tf_audio = tf_audio.astype(np.int16)

    if os.path.exists("./silence") is False:
        os.mkdir("./silence")
    wavfile.write("silence/s_"+str(sample)+".wav", new_sample_rate, tf_audio)

stream.stop_stream()
stream.close()
audio_stream.terminate()