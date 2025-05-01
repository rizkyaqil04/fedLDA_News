import time
import threading
from mqtt_handler import setup_mqtt, close_registration, get_update_round, is_training_done
from mlflow_utils import start_mlflow_run, end_run

if __name__ == "__main__":
    start_mlflow_run(role="server")
    mqtt_client = setup_mqtt()
    print("[SERVER] Registration open for 30 seconds...")
    threading.Thread(target=close_registration, args=(mqtt_client,)).start()

    while not is_training_done():
        time.sleep(1)

    end_run()
    print(f"[SERVER] Training finished after {get_update_round()} rounds.")

