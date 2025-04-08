from src.datasets.mqtt import MQTT
from src.datasets.adult_income import Adult

from types import SimpleNamespace


args = {
    'data_path': '/scratch/wz1232/network-intrusion-detection/datasets/MQTTset/Data/FINAL_CSV',
}

args = SimpleNamespace(**args)

mqtt = MQTT(args)

mqtt.load()

