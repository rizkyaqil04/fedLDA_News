import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import random

# 1. Load dataset
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data[:1000]  # pakai 1000 dokumen saja untuk lebih cepat

# 2. Preprocessing: ubah jadi bag-of-words
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(docs)
vocab = vectorizer.get_feature_names_out()
word2id = {word: idx for idx, word in enumerate(vocab)}
id2word = {idx: word for word, idx in word2id.items()}
W, D = X.shape[1], X.shape[0]

# 3. Konversi dokumen ke daftar kata (list of word indices)
doc_word_ids = []
for d in range(X.shape[0]):
    word_ids = []
    for w_id in X[d].nonzero()[1]:
        count = X[d, w_id]
        word_ids.extend([w_id] * count)
    doc_word_ids.append(word_ids)

# 4. Inisialisasi parameter
K = 10  # jumlah topik
alpha = 0.1
beta = 0.01
iterations = 1000

# Count matrices
n_dk = np.zeros((D, K))       # jumlah topik per dokumen
n_kw = np.zeros((K, W))       # jumlah kata per topik
n_k = np.zeros(K)             # total kata per topik
z_dn = []                     # topik untuk tiap kata di dokumen

# 5. Inisialisasi topik acak untuk tiap kata
for d, word_ids in enumerate(doc_word_ids):
    z_n = []
    for w in word_ids:
        z = random.randint(0, K-1)
        z_n.append(z)
        n_dk[d, z] += 1
        n_kw[z, w] += 1
        n_k[z] += 1
    z_dn.append(z_n)

# 6. Gibbs sampling
for it in tqdm(range(iterations)):
    for d, word_ids in enumerate(doc_word_ids):
        for i, w in enumerate(word_ids):
            z = z_dn[d][i]

            # kurangi count
            n_dk[d, z] -= 1
            n_kw[z, w] -= 1
            n_k[z] -= 1

            # hitung distribusi probabilitas topik
            p_z = (n_kw[:, w] + beta) / (n_k + W * beta) * \
                  (n_dk[d] + alpha)

            p_z = p_z / p_z.sum()

            # sampling topik baru
            new_z = np.random.choice(K, p=p_z)
            z_dn[d][i] = new_z

            # tambahkan count baru
            n_dk[d, new_z] += 1
            n_kw[new_z, w] += 1
            n_k[new_z] += 1

# 7. Tampilkan top-N kata untuk setiap topik
top_n = 10
for k in range(K):
    top_words = np.argsort(n_kw[k])[::-1][:top_n]
    top_terms = [id2word[i] for i in top_words]
    print(f"Topik {k+1}: {', '.join(top_terms)}")

