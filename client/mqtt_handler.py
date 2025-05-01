import paho.mqtt.client as mqtt
import numpy as np
import json
import uuid
import yaml
from client import Client
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# === Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

cfg = config["client"]
global_cfg = config["global"]

# === Konfigurasi dari file
EPSILON = cfg["epsilon"]
ALPHA = cfg["alpha"]
BETA = cfg["beta"]
DELTA = cfg["delta"]
TOPICS = global_cfg["num_topics"]
VOCAB_SIZE = global_cfg["vocab_size"]
DATA_SIZE = cfg["data_size"]
LOCAL_ITERATIONS = cfg["local_iterations"]

# === MQTT
TOPIC_REGISTER = "fedlda/registration"
TOPIC_ACK_BASE = "fedlda/ack/"
TOPIC_UPDATE = "fedlda/update"
TOPIC_PHI = "fedlda/global_phi"

client_uuid = str(uuid.uuid4())
client_id = None
_registered = False

# === Siapkan data & Client LDA
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
vectorizer = CountVectorizer(max_features=VOCAB_SIZE, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)
docs = X.toarray()

doc_idxs = np.random.choice(X.shape[0], size=DATA_SIZE, replace=False)
bow_vec = np.sum(X[doc_idxs].toarray(), axis=0)
doc = []
for word_id, count in enumerate(bow_vec):
    doc.extend([word_id] * count)
np.random.shuffle(doc)

fed_client = Client(doc, TOPICS, VOCAB_SIZE, ALPHA, BETA)

# === Callback Functions
def on_connect(client, userdata, flags, rc):
    print(f"[CLIENT {client_uuid[:8]}] Connected to broker.")
    client.subscribe(TOPIC_ACK_BASE + client_uuid)
    client.subscribe(TOPIC_PHI)
    client.publish(TOPIC_REGISTER, json.dumps({"uuid": client_uuid}))

def on_ack(client, userdata, msg):
    global client_id, _registered
    payload = json.loads(msg.payload.decode())

    if payload.get("status") == "rejected":
        print(f"[CLIENT {client_uuid[:8]}] Registration rejected: {payload['reason']}")
        client.loop_stop()
        return

    client_id = str(payload["client_id"])
    _registered = True
    print(f"[CLIENT {client_uuid[:8]}] Assigned client_id: {client_id}")

def on_phi(client, userdata, msg):
    if client_id is None:
        return

    data = json.loads(msg.payload.decode())
    round_id = data["round"]
    phi = np.array(data["phi"])

    print(f"[Client {client_id}] Received phi for round {round_id}")
    counts = fed_client.gibbs_sampling(phi, iterations=LOCAL_ITERATIONS)
    update = fed_client.perturb_update(counts, epsilon=EPSILON, delta=DELTA)

    client.publish(TOPIC_UPDATE, json.dumps({
        "client_id": client_id,
        "update": update.tolist()
    }))
    print(f"[Client {client_id}] Sent update for round {round_id}")

# === MQTT Client Setup
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.message_callback_add(TOPIC_ACK_BASE + client_uuid, on_ack)
mqtt_client.message_callback_add(TOPIC_PHI, on_phi)

# === Status API
def is_registered():
    return _registered

