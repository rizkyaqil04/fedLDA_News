import time
from mqtt_handler import mqtt_client, is_registered
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    BROKER = config["global"]["broker"]
    PORT = config["global"]["port"]

    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()

    print("[CLIENT] Waiting for registration response...")
    while not is_registered():
        time.sleep(0.5)

    print("[CLIENT] Listening for phi...")
    while True:
        time.sleep(1)

