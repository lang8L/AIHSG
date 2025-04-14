import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from tqdm import tqdm

def main():
 
    print("正在加载数据...")
    try:
        data = pd.read_csv('st_data.csv')
        if 'target' not in data.columns or 'text' not in data.columns:
            raise ValueError("CSV文件必须包含'target'和'text'列")
        print(f"数据加载成功，共 {len(data)} 条记录")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    
    print("\n开始聚类处理...")
    results = []
    
  
    for label in tqdm(data['target'].unique(), desc="处理进度"):
        # 获取当前标签的所有文本
        subset = data[data['target'] == label].copy()
        texts = subset['text'].astype(str).tolist() 
        
      
        if len(texts) < 2:
            subset['internal_cluster'] = 0
            results.append(subset)
            continue
            
       
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  # 包含二元词组
        )
        try:
            X = vectorizer.fit_transform(texts)
            dist_matrix = cosine_distances(X)
        except Exception as e:
            print(f"标签 '{label}' 向量化失败: {str(e)}")
            continue
            
      
        dbscan = DBSCAN(
            eps=0.6,         
            min_samples=2,    
            metric='precomputed',
            n_jobs=-1       
        )
        clusters = dbscan.fit_predict(dist_matrix)
        
       
        clusters[clusters == -1] = np.max(clusters) + 1 if -1 in clusters else 0
        
       
        unique_clusters = np.unique(clusters)
        cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
        subset['internal_cluster'] = [cluster_mapping[c] for c in clusters]
        
        results.append(subset)

   
    final_result = pd.concat(results)
    output_file = 'dbscan_data.csv'
    final_result.to_csv(output_file, index=False)
    
    print(f"\n聚类完成！结果已保存到 {output_file}")
    print("\n聚类结果统计：")
    print(final_result.groupby(['target', 'internal_cluster']).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()