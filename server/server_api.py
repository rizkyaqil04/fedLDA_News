from fastapi import FastAPI, Request
import numpy as np
from server.server import Server
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import json
import os

app = FastAPI()

NUM_CLIENTS = 3
REQUIRED_UPDATES = NUM_CLIENTS
MAX_ROUNDS = 10
UPDATE_ROUND = 0
VOCAB_SIZE = 10000

server = Server(num_topics=5, vocab_size=VOCAB_SIZE, beta=0.01)
update_buffer = []
clients_updated = set()
phi_history = []

# Testing vocab (dari 20newsgroups)
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes')).data
vectorizer = CountVectorizer(max_features=VOCAB_SIZE, stop_words='english')
vectorizer.fit(test_data)
vocab = vectorizer.get_feature_names_out()

def extract_top_words(phi, vocab, top_n=5):
    results = []
    for topic_id, topic_dist in enumerate(phi):
        top_idx = np.argsort(topic_dist)[::-1][:top_n]
        words = [vocab[i] for i in top_idx]
        results.append((topic_id, words))
    return results

@app.get("/get-phi/{client_id}")
async def get_phi(client_id: int):
    if UPDATE_ROUND >= MAX_ROUNDS:
        return {"status": "done", "phi": None, "round": UPDATE_ROUND}

    if client_id in clients_updated:
        return {"status": "wait", "phi": None, "round": UPDATE_ROUND}

    return {"status": "ok", "phi": server.phi.tolist(), "round": UPDATE_ROUND}

@app.post("/update")
async def post_update(request: Request):
    global UPDATE_ROUND

    if UPDATE_ROUND >= MAX_ROUNDS:
        return {"status": "max_round_reached"}

    data = await request.json()
    client_id = data["client_id"]
    update_matrix = np.array(data["update"])
    update_buffer.append(update_matrix)
    clients_updated.add(client_id)

    if len(update_buffer) == REQUIRED_UPDATES:
        server.aggregate_updates(update_buffer)
        phi_history.append(server.phi.copy())

        # Evaluasi coherence
        texts = [[w for w in doc.lower().split() if w in set(vocab)] for doc in test_data]
        coherence = server.evaluate_coherence(server.phi, vocab, texts)
        print(f"[SERVER] Coherence Score (c_v): {coherence:.4f}")

        update_buffer.clear()
        clients_updated.clear()

        print(f"\n[SERVER] Aggregated update for round {UPDATE_ROUND}")
        top_words = extract_top_words(server.phi, vocab, top_n=5)
        for tid, words in top_words:
            print(f"  Topic {tid}: {', '.join(words)}")

        os.makedirs("phi_logs", exist_ok=True)
        with open(f"phi_logs/phi_round_{UPDATE_ROUND}.json", "w") as f:
            json.dump(server.phi.tolist(), f)

        UPDATE_ROUND += 1

    return {"status": "received"}

@app.get("/round")
async def get_current_round():
    return {"current_round": UPDATE_ROUND}

@app.get("/phi-log")
async def get_phi_log():
    return {"history": [phi.tolist() for phi in phi_history]}

