import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('st_data.csv') 

results = []

for label in data['target'].unique():
    print(f"\nProcessing label: {label}")
    subset = data[data['target'] == label].copy()
    texts = subset['text'].tolist()
    
  
    if len(texts) < 2:
        subset['internal_cluster'] = f"{label}_cluster_0"
        results.append(subset)
        continue
    
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)  
    )
    X = vectorizer.fit_transform(texts)
    
  
    agg = AgglomerativeClustering(
        n_clusters=None,         
        metric='cosine',         
        linkage='average',       
        distance_threshold=0.9  
    )
    clusters = agg.fit_predict(X.toarray())
    
   
    unique_clusters = np.unique(clusters)
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
    subset['internal_cluster'] = [f"{label}_cluster_{cluster_mapping[c]}" for c in clusters]
    
   
    n_clusters = len(unique_clusters)
    print(f"生成 {n_clusters} 个内部簇")
    print(pd.Series(subset['internal_cluster']).value_counts())
    
   
    Z = linkage(X.toarray(), method='average', metric='cosine')
    plt.figure(figsize=(10, 5))
    plt.title(f'Dendrogram for {label} (Distance Threshold=0.6)')
    dendrogram(Z, truncate_mode='level', p=3)
    plt.axhline(y=0.6, c='k', ls='--')
    plt.show()
    
    results.append(subset)


final_result = pd.concat(results)
final_result.to_csv('agnes_data.csv', index=False)

print("\n聚类完成！结果列表示例：")
print(final_result[['target', 'text', 'internal_cluster']].head())