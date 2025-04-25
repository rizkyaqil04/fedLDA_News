import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Fetch data tanpa menghapus bagian header, footer, dan quotes
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Menampilkan data mentah dalam bentuk DataFrame
df_raw = pd.DataFrame(newsgroups.data, columns=['Text'])

# Menampilkan beberapa baris pertama
print(df_raw)

