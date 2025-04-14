import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
import numpy as np

# 加载数据
data = pd.read_csv('st_data.csv')  # 确保包含target和text列

# 对每个target标签内部独立聚类
results = []

for label in data['target'].unique():
    print(f"\nProcessing label: {label}")
    subset = data[data['target'] == label].copy()
    texts = subset['text'].tolist()
    
    # 样本数不足时处理
    if len(texts) < 2:
        subset['internal_cluster'] = f"{label}_cluster_0"
        results.append(subset)
        continue
    
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(
        max_features=500,       # 减少特征维度以适应GMM
        stop_words='english',
        ngram_range=(1, 1)     # GMM更适合单字特征
    )
    X = vectorizer.fit_transform(texts)
    
    # 降维处理（GMM需要稠密矩阵且性能敏感）
    svd = TruncatedSVD(n_components=50)  # 降至50维
    X_reduced = svd.fit_transform(X)
    
    # GMM聚类
    gmm = GaussianMixture(
        n_components=5,          # 每个标签内部预设3个簇（可调整）
        covariance_type='diag',  # 对角协方差（适合文本数据）
        random_state=42
    )
    clusters = gmm.fit_predict(X_reduced)
    
    # 添加簇信息
    subset['internal_cluster'] = [f"{label}_cluster_{c}" for c in clusters]
    
    # 打印当前标签聚类情况
    n_clusters = len(np.unique(clusters))
    print(f"生成 {n_clusters} 个内部簇")
    print(pd.Series(subset['internal_cluster']).value_counts())
    
    results.append(subset)

# 合并保存结果
final_result = pd.concat(results)
final_result.to_csv('gmm_data.csv', index=False)

print("\n聚类完成！结果列表示例：")
print(final_result[['target', 'text', 'internal_cluster']].head(10))