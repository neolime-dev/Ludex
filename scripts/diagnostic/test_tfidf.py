import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/processed/games.csv')
matches = df[df['title'].astype(str).str.contains('Minecraft:? Dungeons', case=False, na=False, regex=True)]
if matches.empty:
    raise RuntimeError("Minecraft Dungeons nao encontrado no dataset.")
mc_idx = matches.index[0]

text = df['title'].fillna('') + " " + (df['genres'].fillna('') + " ") * 3 + df['description'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=30000)
matrix = vectorizer.fit_transform(text)

feature_names = vectorizer.get_feature_names_out()
vector = matrix[mc_idx].toarray().flatten()
top_indices = vector.argsort()[::-1][:10]

print(f"Top TF-IDF features for {df.loc[mc_idx, 'title']}:")
for idx in top_indices:
    print(f"{feature_names[idx]}: {vector[idx]:.4f}")
