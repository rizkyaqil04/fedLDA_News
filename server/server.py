import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

class Server:
    def __init__(self, num_topics, vocab_size, beta):
        self.K = num_topics
        self.V = vocab_size
        self.beta = beta
        self.phi = np.random.dirichlet([beta] * vocab_size, num_topics)  # K x V

    def aggregate_updates(self, updates):
        summed = np.sum(updates, axis=0).astype(np.float64)
        # summed = np.clip(summed, 0, None)
        summed += self.beta

        row_sums = np.sum(summed, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero

        self.phi = summed / row_sums

    def evaluate_coherence(self, phi, vocab, texts, top_n=10, coherence_type='c_v'):
        # Ambil top-N kata per topik berdasarkan phi
        topics = []
        for topic_dist in phi:
            top_indices = topic_dist.argsort()[::-1][:top_n]
            top_words = [vocab[i] for i in top_indices]
            topics.append(top_words)

        # Siapkan dictionary
        dictionary = Dictionary(texts)

        # Hitung coherence
        cm = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            coherence=coherence_type
        )
        return cm.get_coherence()
