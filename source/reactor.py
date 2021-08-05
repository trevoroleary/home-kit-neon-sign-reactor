import logging
import pyaudio
import numpy as np
import time
import wave
from gpiozero import PWMLED
from time import sleep
import _thread
import pickle

LED_PINS = [18, 12, 19]
leds = [PWMLED(pin, frequency=100, active_high=False) for pin in LED_PINS]
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = logging.FileHandler('data_out.log')
log_file.setLevel(logging.DEBUG)
logger.addHandler(log_file)

# FFT Moving Average Settings
WINDOW_SIZE = 4
LEVELS = [0]*WINDOW_SIZE # Window of brightness levels for neon

# Audio Stream Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # The other mic works at 44100
CHUNK = 1024 # RATE / number of updates per second
window = np.blackman(CHUNK) # Decaying window on either side of chunk

# Debug Variables
SOUND_DATA = list()

def set_level(bass_level):
    if bass_level > 1:
        bass_level = 1
    for led in leds:
        led.value = bass_level

def pulse():
    for i in range(100, 0, -1):
        for led in leds:
            led.value = i/100
        sleep(0.005)
    led.off()

def old_reactive_calc(fft_data):
    trigger_thresh = 0.3e6
    sense_range = fft_data[1:25]
    if any(y > trigger_thresh for y in sense_range):
        _thread.start_new_thread(pulse, ())

def volume_reactive_calc(fft_data):
    global LEVELS
    high_thresh = 0.3e6
    low_thresh = 0.1e6
    sense_range = fft_data[1:10]
    max_all = max(sense_range)
    if max_all > low_thresh:
        level_append = (max_all - low_thresh)/(high_thresh - low_thresh)
    else:
        level_append = 0
    LEVELS.append(level_append/WINDOW_SIZE)
    LEVELS.pop(0)
    logger.info(f"Sum of Levels is {sum(LEVELS)}")
    set_level(sum(LEVELS))

def instant_volume_reactive_calc(fft_data):
    global LEVELS
    # Constants
    high_thresh = 0.3e6
    low_thresh = 0.05e6
    rest_level = 0.02

    sense_range = fft_data[1:10]
    max_all = max(sense_range)
    offset = rest_level * (1 - (max_all - low_thresh) / high_thresh)
    if max_all > high_thresh:
        LEVELS = [(max_all - low_thresh)/(WINDOW_SIZE*(high_thresh - low_thresh)) + offset]*WINDOW_SIZE
        _thread.start_new_thread(pulse, ())
    if max_all > low_thresh:
        LEVELS.append((max_all - low_thresh)/(WINDOW_SIZE*(high_thresh - low_thresh)) + offset)
        LEVELS.pop(0)
    else:
        LEVELS.append(rest_level)
        LEVELS.pop(0)
    _thread.start_new_thread(set_level, (sum(LEVELS),))

def soundPlot(stream):
    t1 = time.time()
    data = stream.read(CHUNK, exception_on_overflow=False)
    waveData = wave.struct.unpack("%dh"%(CHUNK), data)
    npArrayData = np.array(waveData)
    indata = npArrayData*window

    fftData=np.abs(np.fft.rfft(indata))
    # fftTime=np.fft.rfftfreq(CHUNK, 1./RATE)

    # SOUND_DATA.append(fftData)

    # Reaction Types
    # old_reactive_calc(fftData)
    # volume_reactive_calc(fftData)
    instant_volume_reactive_calc(fftData)


# p=pyaudio.PyAudio()
# stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                # frames_per_buffer=CHUNK, input_device_index=2)

# plt.ion()
# fig = plt.figure(figsize=(10, 8))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):


# stream.stop_stream()
# stream.close()
# p.terminate()
# sound_file = open('sound_data.pickle', 'wb')
# pickle.dump(SOUND_DATA, sound_file)
# sound_file.close()
