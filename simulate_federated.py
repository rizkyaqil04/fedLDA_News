import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from server import Server
from client import Client

# Konfigurasi
NUM_CLIENTS = 5
NUM_TOPICS = 5
ALPHA = 0.1
BETA = 0.01
DELTA = 0.1
ITERATIONS = 10
TOP_K = 10

# === Load Dataset ===
print("Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)
vocab = vectorizer.get_feature_names_out()
VOCAB_SIZE = len(vocab)

# Bagi dokumen ke klien
docs_by_client = np.array_split(X.toarray(), NUM_CLIENTS)

# === Utility Functions ===
def extract_top_words(phi, vocab, top_n=5):
    top_words = []
    for topic_id, topic_dist in enumerate(phi):
        top_indices = np.argsort(topic_dist)[::-1][:top_n]
        words = [vocab[i] for i in top_indices]
        top_words.append((topic_id, words))
    return top_words

def compute_perplexity(clients, phi):
    total_log_prob = 0
    total_words = 0
    for client in clients:
        for word in client.doc:
            topic_probs = client.theta * phi[:, word]
            word_prob = np.sum(topic_probs)
            word_prob = max(word_prob, 1e-12)
            total_log_prob += np.log(word_prob)
            total_words += 1
    return np.exp(-total_log_prob / total_words)

def visualize_wordclouds(phi, vocab):
    for topic_id, topic_dist in enumerate(phi):
        word_freq = {vocab[i]: float(topic_dist[i]) for i in range(len(vocab))}
        wc = WordCloud(width=500, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {topic_id}")
    plt.tight_layout()
    plt.show()

def save_results_json(phi, vocab, eps, perplexity):
    result = {
        "epsilon": eps,
        "perplexity": perplexity,
        "topics": []
    }
    for topic_id, topic_dist in enumerate(phi):
        sorted_indices = np.argsort(topic_dist)[::-1]
        words_probs = [{"word": vocab[i], "prob": float(topic_dist[i])} for i in sorted_indices[:TOP_K]]
        result["topics"].append({
            "topic_id": topic_id,
            "top_words": words_probs
        })
    with open(f"fedlda_result_eps{eps}.json", "w") as f:
        json.dump(result, f, indent=2)

# === Eksperimen Multi-ε ===
epsilons = [0.1, 0.5, 1, 2, 5, 10]
results = []

for eps in epsilons:
    print(f"\n=== Training with ε = {eps} ===")
    server = Server(NUM_TOPICS, VOCAB_SIZE, BETA)

    clients = []
    for doc_group in docs_by_client:
        for doc_vec in doc_group:
            doc = []
            for word_id, count in enumerate(doc_vec):
                doc += [word_id] * count
            if len(doc) > 0:
                np.random.shuffle(doc)
                clients.append(Client(doc, NUM_TOPICS, VOCAB_SIZE, ALPHA, BETA))

    for t in range(ITERATIONS):
        updates = []
        for client in clients:
            local_counts = client.gibbs_sampling(server.phi)
            perturbed = client.perturb_update(local_counts, eps, delta=DELTA)
            updates.append(perturbed)
        server.aggregate_updates(updates)

    phi = server.phi
    perplexity = compute_perplexity(clients, phi)
    top_words = extract_top_words(phi, vocab, top_n=TOP_K)

    print(f"Perplexity: {perplexity:.2f}")
    for topic_id, words in top_words:
        print(f"Topic {topic_id}: {words}")

    save_results_json(phi, vocab, eps, perplexity)
    visualize_wordclouds(phi, vocab)
    results.append((eps, perplexity))

# === Plot ε vs Perplexity ===
eps_vals, perp_vals = zip(*results)
plt.figure()
plt.plot(eps_vals, perp_vals, marker='o')
plt.xlabel("Privacy Budget ε")
plt.ylabel("Perplexity")
plt.title("Effect of ε on Model Accuracy (Perplexity)")
plt.grid(True)
plt.tight_layout()
plt.savefig("perplexity_vs_epsilon.png")
plt.show()

