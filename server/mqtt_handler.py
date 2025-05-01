import json
import time
import os
import numpy as np
import threading
import paho.mqtt.client as mqtt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from server import Server
from mlflow_utils import log_coherence, log_phi_top_words
import yaml

# === Load config.yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Konstanta dari config
VOCAB_SIZE = config["global"]["vocab_size"]
NUM_TOPICS = config["global"]["num_topics"]
MAX_ROUNDS = config["global"]["max_rounds"]
NUM_CLIENTS = config["global"]["num_clients"]
REGISTRATION_WINDOW = config["server"]["registration_window"]
BETA = config["server"]["beta"]
BROKER = config["global"]["broker"]
PORT = config["global"]["port"]

# === MQTT Topics
TOPIC_REGISTER = "fedlda/registration"
TOPIC_ACK_BASE = "fedlda/ack/"
TOPIC_UPDATE = "fedlda/update"
TOPIC_GLOBAL_PHI = "fedlda/global_phi"

# === Global State
UPDATE_ROUND = 0
server = Server(num_topics=NUM_TOPICS, vocab_size=VOCAB_SIZE, beta=BETA)
if server.load_phi("phi_logs/phi_latest.json"):
    print("[SERVER] Loaded previous phi model from disk.")
else:
    print("[SERVER] Initialized new phi model.")

registered_clients = {}
client_counter = 0
registration_open = True
update_buffer = []
clients_updated = set()
phi_history = []

# === Dataset Evaluasi
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes')).data
vectorizer = CountVectorizer(max_features=VOCAB_SIZE, stop_words='english')
vectorizer.fit(test_data)
vocab = vectorizer.get_feature_names_out()

# === Setup MQTT
def setup_mqtt():
    client = mqtt.Client()
    client.on_connect = lambda c, u, f, rc: print("[SERVER] Connected to broker")
    client.connect(BROKER, PORT, 60)
    client.subscribe(TOPIC_REGISTER)
    client.subscribe(TOPIC_UPDATE)
    client.message_callback_add(TOPIC_REGISTER, on_registration)
    client.message_callback_add(TOPIC_UPDATE, on_update)
    client.loop_start()
    return client

def close_registration(client):
    global registration_open
    time.sleep(REGISTRATION_WINDOW)
    registration_open = False
    print(f"[SERVER] Registration closed with {len(registered_clients)} clients.")

    time.sleep(2)
    client.publish(TOPIC_GLOBAL_PHI, json.dumps({
        "phi": server.phi.tolist(),
        "round": UPDATE_ROUND
    }))

# === Callback MQTT
def on_registration(client, userdata, msg):
    global client_counter
    if not registration_open:
        return

    data = json.loads(msg.payload.decode())
    uuid = data["uuid"]

    if uuid in registered_clients:
        return

    if len(registered_clients) >= NUM_CLIENTS:
        client.publish(TOPIC_ACK_BASE + uuid, json.dumps({
            "status": "rejected",
            "reason": "max_clients_reached"
        }))
        print(f"[SERVER] Rejected {uuid[:8]}: max clients reached")
        return

    assigned_id = str(client_counter)
    registered_clients[uuid] = assigned_id
    client_counter += 1
    print(f"[SERVER] Registered {uuid[:8]} as client_id {assigned_id}")

    client.publish(TOPIC_ACK_BASE + uuid, json.dumps({
        "status": "accepted",
        "client_id": assigned_id
    }))

def on_update(client, userdata, msg):
    global UPDATE_ROUND
    if UPDATE_ROUND >= MAX_ROUNDS:
        return

    data = json.loads(msg.payload.decode())
    client_id = str(data["client_id"])

    if client_id not in registered_clients.values():
        print(f"[SERVER] Ignored update from unknown client_id {client_id}")
        return

    update_matrix = np.array(data["update"])
    update_buffer.append(update_matrix)
    clients_updated.add(client_id)
    print(f"[SERVER] Received update from client {client_id}")

    if len(clients_updated) == len(registered_clients):
        server.aggregate_updates(update_buffer)
        phi_history.append(server.phi.copy())

        texts = [[w for w in doc.lower().split() if w in set(vocab)] for doc in test_data]
        coherence = server.evaluate_coherence(server.phi, vocab, texts)
        print(f"[SERVER] Coherence Score (c_v): {coherence:.4f}")

        update_buffer.clear()
        clients_updated.clear()

        print(f"[SERVER] Aggregated Round {UPDATE_ROUND}")
        top_words = extract_top_words(server.phi, vocab)
        for tid, words in top_words:
            print(f"  Topic {tid}: {', '.join(words)}")

        os.makedirs("phi_logs", exist_ok=True)
        with open(f"phi_logs/phi_round_{UPDATE_ROUND}.json", "w") as f:
            json.dump(server.phi.tolist(), f)

        server.save_phi("phi_logs/phi_latest.json")  # ✅ Save model for reuse

        log_coherence(coherence, UPDATE_ROUND)
        log_phi_top_words(top_words, UPDATE_ROUND)

        UPDATE_ROUND += 1

        if UPDATE_ROUND < MAX_ROUNDS:
            client.publish(TOPIC_GLOBAL_PHI, json.dumps({
                "phi": server.phi.tolist(),
                "round": UPDATE_ROUND
            }))
        else:
            print("[SERVER] Max rounds reached. Training complete.")

def extract_top_words(phi, vocab, top_n=5):
    return [(tid, [vocab[i] for i in np.argsort(phi[tid])[::-1][:top_n]]) for tid in range(len(phi))]

# === Status API
def get_update_round():
    return UPDATE_ROUND

def is_training_done():
    return UPDATE_ROUND >= MAX_ROUNDS

