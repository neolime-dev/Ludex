import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/processed/games.csv')
mc_idx = df[df['title'] == 'Minecraft Dungeons'].index[0]

text = df['title'].fillna('') + " " + (df['genres'].fillna('') + " ") * 3 + df['description'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=30000)
matrix = vectorizer.fit_transform(text)

feature_names = vectorizer.get_feature_names_out()
vector = matrix[mc_idx].toarray().flatten()
top_indices = vector.argsort()[::-1][:10]

print("Top TF-IDF features for Minecraft Dungeons:")
for idx in top_indices:
    print(f"{feature_names[idx]}: {vector[idx]:.4f}")
