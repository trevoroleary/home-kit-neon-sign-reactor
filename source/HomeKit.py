"""Starts a fake fan, lightbulb, garage door and a TemperatureSensor
"""
import logging
import requests
from bs4 import BeautifulSoup
import signal
from numpy.fft import fft
import pyaudio
from pyhap.service import Service
import wave
import numpy as np
from time import sleep, perf_counter
import _thread
from pyhap.accessory import Accessory, Bridge
from gpiozero import PWMLED
from pyhap.accessory_driver import AccessoryDriver
from pyhap.const import (CATEGORY_FAN,
                         CATEGORY_LIGHTBULB,
                         CATEGORY_GARAGE_DOOR_OPENER,
                         CATEGORY_SENSOR)

logging.basicConfig(level=logging.INFO, format="[%(module)s] %(message)s")

class HomeKitNeonLight(Accessory):
    """Fake lightbulb, logs what the client sets."""

    category = CATEGORY_LIGHTBULB

    def __init__(self, leds, *args, **kwargs):
        self.leds = leds
        self.leds = [PWMLED(pin, frequency=100, active_high=False) for pin in leds]
        super().__init__(*args, **kwargs)

        serv_light = self.add_preload_service('Lightbulb', chars=['On', 'Brightness'])

        self.char_on = serv_light.configure_char('On', setter_callback=self.set_on_off)
        self.char_brightness = serv_light.configure_char('Brightness', setter_callback=self.set_brightness)

    def pulse(self):
        for i in range(100, 50, -1):
            for led in self.leds:
                led.value = i/100
            sleep(0.005)
        # led.off()

    def set_on_off(self, value):
        if value == 0:
            self.set_level_all_ramp(0)
            logging.info("Turned Off Neon")
        else:
            self.set_level_all_ramp(self.char_brightness.value)
    
    def set_level_instant(self, level):
        if level > 1:
            level = 1
        for led in self.leds:
            led.value = level

    def set_level_all_ramp(self, level):
        """
        Takes value from 0->100 as float
        """
        for led in self.leds:
            _thread.start_new_thread(self.set_level_one_ramp, (led, level), {})
    
    def set_level_one_ramp(self, led, level):
        old_value = led.value*100
        direction = 1 if level > old_value else -1
        for intermediate_level in range(int(old_value), int(level), direction):
            led.value = intermediate_level/100
            sleep(0.007)
    
    def set_brightness(self, value):
        self.set_level_all_ramp(value)
        logging.info(f"Neon Value Set to {value}%")

class LedController(Accessory):

    category = CATEGORY_LIGHTBULB
    RATE = 16000        # The other mic works at 44100
    CHUNK = 1024        # RATE / number of updates per second
    BLACKMAN_WINDOW = np.blackman(CHUNK) # Decaying window on either side of chunk

    HIGH_THRESH_MAX = 0.7e6
    LOW_THRESH_MAX = 0.05e6

    MOVING_WINDOW_SIZE = 4
    MOVING_WINDOW = [0] * MOVING_WINDOW_SIZE

    def __init__(self, lights, *args, **kwargs):
        self.lights: list = lights
        self.logger = logging.getLogger(__name__)
        super().__init__(*args, **kwargs)

        self.last_temp_time = perf_counter()

        serv_light = self.add_preload_service('Lightbulb', chars=['On', 'Brightness'])
        self.char_on = serv_light.configure_char('On', setter_callback=self.set_on_off)
        self.char_brightness = serv_light.configure_char('Brightness', setter_callback=self.set_thresholds)
        self.set_up_stream()

        self.high_thresh = 0.3e6
        self.low_thresh = 0.05e6
        self.rest_level = 0.1
        self.span = self.high_thresh - self.low_thresh
        self.offset = self.rest_level * self.span + self.low_thresh

    def set_up_stream(self):
        p=pyaudio.PyAudio()
        self.stream=p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK, 
            input_device_index=2)

    """ ----------------- Important Accessory Functions ----------------- """
    def set_on_off(self, value):
        if value:
            self.logger.info("Turning ON Reactor")
        else:
            self.logger.info("Turning OFF Reactor")
        pass
    
    def set_thresholds(self, value):
        self.high_thresh = self.HIGH_THRESH_MAX * value / 100
        # self.low_thresh = self.LOW_THRESH_MAX * value / 100
        self.span = self.high_thresh - self.low_thresh
        self.offset = self.rest_level * self.span + self.low_thresh
        
        # self.char_brightness = value
        logging.info(f"Neon reactor thresholds set to {value}% of max")

    """ ----------------- Runs at Internal ----------------- """
    @Accessory.run_at_interval(0.033)
    def run(self):
        if self.char_on.value == 1:
            fft_data = self.get_fft_data()
            self.instant_volume_reactive_calc(fft_data=fft_data)
        else:
            for light in self.lights:
                if light.char_on.value == 1 or light.char_brightness.value == 0:
                    light.set_level_all_ramp(light.char_brightness.value)
                elif light.char_on.value == 0:
                    light.set_level_all_ramp(0)
        
    """ ----------------- FFT ----------------- """
    def get_fft_data(self):
        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        waveData = wave.struct.unpack("%dh"%(self.CHUNK), data)
        npArrayData = np.array(waveData)
        indata = npArrayData*self.BLACKMAN_WINDOW

        fft_data=np.abs(np.fft.rfft(indata))
        return fft_data
    
    """ ----------------- Reactor ----------------- """
    def instant_volume_reactive_calc(self, fft_data):
        sense_range = fft_data[1:10]
        max_all = max(sense_range)

        if max_all > self.high_thresh:
            self.MOVING_WINDOW = [1]*self.MOVING_WINDOW_SIZE
            # for light in self.lights:
                # _thread.start_new_thread(light.pulse, ())
        if max_all > (self.low_thresh + self.offset):
            percent = (max_all - self.low_thresh)/self.span
            self.MOVING_WINDOW.append(percent)
            self.MOVING_WINDOW.pop(0)
        else:
            self.MOVING_WINDOW.append(self.rest_level)
            self.MOVING_WINDOW.pop(0)
        for light in self.lights:
            _thread.start_new_thread(light.set_level_instant, (sum(self.MOVING_WINDOW)/self.MOVING_WINDOW_SIZE,))

class AirQualitySensor(Accessory):

    category = CATEGORY_SENSOR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        serv_temp = self.add_preload_service('AirQualitySensor')
        self.char_temp = serv_temp.configure_char('AirQuality', setter_callback=self.get_air_quality1)
    @Accessory.run_at_interval(3)
    def run(self):
        self.char_temp.set_value(1)
    
    def get_air_quality1(self):
        self.char_temp.set_value(1)
        return 1

    def get_air_quality(self) -> int:
        return 1
        try:
            x2 = requests.get("https://www.iqair.com/us/usa/california/san-francisco")
            soup = BeautifulSoup(x2.text, 'html.parser')
            air_quality = str(soup.find(class_="aqi-overview-detail__main-pollution-table").next_element.nextSibling.contents[0].contents[1].next)
            air_quality = 1
            return air_quality
        except Exception:
            return 0.0


def get_bridge(driver):
    bridge = Bridge(driver, 'Bridge')
    
    # Accessories
    light_1 = HomeKitNeonLight([12], driver, 'Neon Face')
    light_2 = HomeKitNeonLight([13], driver, 'Neon Smoking')
    air_quality_sensor = AirQualitySensor(driver, "Air Quality")
    neon_controller = LedController([light_1, light_2], driver, "Reactor")

    # Bridge Association
    bridge.add_accessory(light_1)
    bridge.add_accessory(light_2)
    bridge.add_accessory(neon_controller)
    bridge.add_accessory(air_quality_sensor)

    return bridge


def main():
    driver = AccessoryDriver(port=51826, persist_file='/home/pi/PycharmProjects/HomeKit/PiHomeKit.state')
    driver.add_accessory(accessory=get_bridge(driver))
    signal.signal(signal.SIGTERM, driver.signal_handler)
    driver.start()


if __name__ == "__main__":
    main()