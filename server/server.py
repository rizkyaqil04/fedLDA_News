import numpy as np

class Server:
    def __init__(self, num_topics, vocab_size, beta):
        self.K = num_topics
        self.V = vocab_size
        self.beta = beta
        self.phi = np.random.dirichlet([beta] * vocab_size, num_topics)  # K x V

    def aggregate_updates(self, updates):
        summed = np.sum(updates, axis=0).astype(np.float64)
        summed = np.clip(summed, 0, None)
        summed += self.beta

        row_sums = np.sum(summed, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero

        self.phi = summed / row_sums

