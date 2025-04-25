import requests
import time
import numpy as np
import sys
from client import Client
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

client_id = int(sys.argv[1])
server_url = "http://localhost:8000"
EPSILON = 1.0
ALPHA = 0.1
BETA = 0.01
DELTA = 0.1
TOPICS = 5
DATA_SIZE = 1000
VOCAB_SIZE = 10000
LOCAL_ITERATIONS = 100


# Load data
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

client = Client(doc, TOPICS, VOCAB_SIZE, ALPHA, BETA)

last_round = -1
while True:
    try:
        r = requests.get(f"{server_url}/get-phi/{client_id}").json()
    except Exception as e:
        print(f"[Client {client_id}] Error: {e}")
        time.sleep(2)
        continue

    if r["status"] == "done":
        print(f"[Client {client_id}] Training complete. Max round reached.")
        break

    if r["phi"] is None:
        print(f"[Client {client_id}] Phi not ready. Waiting...")
        time.sleep(2)
        continue

    round_id = r["round"]
    if round_id == last_round:
        print(f"[Client {client_id}] Already updated for round {round_id}, sleeping...")
        time.sleep(2)
        continue

    print(f"[Client {client_id}] Received phi for round {round_id}")
    last_round = round_id

    phi = np.array(r["phi"])
    counts = client.gibbs_sampling(phi, iterations=LOCAL_ITERATIONS)
    update = client.perturb_update(counts, epsilon=EPSILON, delta=DELTA)

    requests.post(f"{server_url}/update", json={
        "client_id": client_id,
        "update": update.tolist()
    })
    print(f"[Client {client_id}] Sent update for round {round_id}")

