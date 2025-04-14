import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('st_data.csv')

targets = data['target']


vectorizer = TfidfVectorizer(max_features=1000) 
X = vectorizer.fit_transform(targets)


kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)


data['cluster'] = clusters


data.to_csv('kmeans_data.csv', index=False)

print("聚类完成，结果已保存到 kmeans_data.csv")