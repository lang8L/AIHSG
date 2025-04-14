import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score


nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return " ".join(lemmatized_words)



data = pd.read_csv('st_data.csv')

data['target'] = data['target'].apply(preprocess_text)


vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['target']).toarray()


silhouette_scores = []
calinski_scores = []
valid_k_values = []
best_kmedoids = None
best_labels = None
best_silhouette_score = -1
max_tries = 30  

for _ in range(max_tries):
    for k in range(2, 11):
        kmedoids = KMedoids(
            n_clusters=k,
            metric='cosine',
            init='k-medoids++', 
            max_iter=300 
        )
        kmedoids.fit(X)
        labels = kmedoids.labels_
        unique_labels = len(set(labels))

        if unique_labels >= 2:
            valid_k_values.append(k)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            calinski_scores.append(calinski_harabasz_score(X, labels))

            if score > best_silhouette_score:
                best_silhouette_score = score
                best_kmedoids = kmedoids
                best_labels = labels


if best_kmedoids:
    best_k = valid_k_values[silhouette_scores.index(max(silhouette_scores))]
    data['cluster'] = best_labels
else:
    print("经过多次尝试，仍无法使用K-Medoids得到有效聚类结果。")



last_column = data.iloc[:, -1]

data['original_last_column'] = last_column

if 'cluster' in data.columns:
    data.to_csv('kmedoids_data.csv', index=False)
    print("K - Medoids聚类完成！")
    print(f"最优聚类数量: {best_k}")
    print("每个簇的样本数量:")
    print(data['cluster'].value_counts().sort_index())
else:
    print("聚类结果未成功生成，无法保存数据。")