import numpy as np

def normalize_or_uniform(prob_vec):
    prob_vec = np.clip(prob_vec, 0, None)
    total = np.sum(prob_vec)
    if total <= 0 or np.isnan(total):
        return np.ones_like(prob_vec) / len(prob_vec)
    return prob_vec / total

def get_truncated_set(phi_k, delta):
    sorted_indices = np.argsort(phi_k)[::-1]
    cumulative = np.cumsum(phi_k[sorted_indices])
    cutoff = np.searchsorted(cumulative, 1 - delta, side='right') + 1
    return set(sorted_indices[:cutoff])

class Client:
    def __init__(self, doc, K, V, alpha, beta):
        self.doc = doc
        self.K = K
        self.V = V
        self.alpha = alpha
        self.beta = beta
        self.N = len(doc)

        self.theta = np.random.dirichlet([alpha] * K)
        self.z = np.random.choice(K, self.N)
        self.last_phi = None

    def gibbs_sampling(self, phi, iterations=1):
        counts = np.zeros((self.K, self.V), dtype=np.int32)
        topic_counts = np.zeros(self.K, dtype=np.int32)

        for _ in range(iterations):
            for i, word in enumerate(self.doc):
                topic = self.z[i]
                topic_counts[topic] -= 1
                counts[topic, word] -= 1

                topic_dist = (topic_counts + self.alpha) * (phi[:, word] + self.beta)
                probs = normalize_or_uniform(topic_dist)

                new_topic = np.random.choice(self.K, p=probs)
                self.z[i] = new_topic
                topic_counts[new_topic] += 1
                counts[new_topic, word] += 1

        self.theta = normalize_or_uniform(topic_counts + self.alpha)
        self.last_phi = phi.copy()
        return counts

    def rrp_perturb(self, w, k1, k2, theta, phi, delta, eta):
        if np.random.rand() > eta:
            return (w, k1, k2)

        p_theta = normalize_or_uniform(theta)
        k_prime = np.random.choice(self.K, p=p_theta)

        word_probs = normalize_or_uniform(phi[k_prime])
        w_prime = np.random.choice(self.V, p=word_probs)

        top_words = get_truncated_set(word_probs, delta)
        if w_prime in top_words:
            return (w_prime, k1, k2)
        else:
            return (w, k1, k2)

    def perturb_update(self, counts, epsilon, delta=0.1):
        eta = 1 / (1 + np.exp(epsilon))
        tuples = []

        for k in range(self.K):
            for v in range(self.V):
                c = counts[k, v]
                if c > 0:
                    for _ in range(c):
                        tuples.append((v, -1, k))

        perturbed = np.zeros((self.K, self.V), dtype=np.float64)
        for (w, k1, k2) in tuples:
            w_perturbed, _, _ = self.rrp_perturb(w, k1, k2, self.theta, self.last_phi, delta, eta)
            perturbed[k2, w_perturbed] += 1

        return perturbed

