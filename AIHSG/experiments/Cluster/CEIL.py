import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# 读取 CSV 文件并转换为 DataFrame
df = pd.read_csv('------', encoding='utf-8')

# 将文本列转换为列表
texts = df['text'].tolist()

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# 提取文本特征
def extract_features(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():  # 使用torch.no_grad()避免计算梯度
        outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # 取CLS标记的输出并移动到CPU
    print("提取文本特征完成")
    return features


# 聚类
def cluster_texts(features, n_clusters=12):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    print("聚类完成")
    return kmeans.labels_


# 生成代表性词汇标签
def generate_labels(texts, cluster_labels, n_clusters=12, top_n_words=3):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit(texts)

    feature_names = vectorizer.get_feature_names_out()

    cluster_words_map = {}
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_tf_idf_matrix = tfidf_matrix.transform(np.array(texts)[cluster_indices]).toarray()
        avg_tf_idf_vector = np.mean(cluster_tf_idf_matrix, axis=0)
        top_n_word_indices = avg_tf_idf_vector.argsort()[-top_n_words:][::-1]
        top_n_words_cluster = [feature_names[index] for index in top_n_word_indices]
        cluster_words_map[i] = ', '.join(top_n_words_cluster)

    # 为每个文本分配其簇的代表性词汇标签
    text_labels = [cluster_words_map[label] for label in cluster_labels]
    print("生成词汇标签完成")
    return text_labels


# 聚类流程
features = extract_features(texts)
cluster_labels = cluster_texts(features)

# 生成标签并导出到CSV
text_labels = generate_labels(texts, cluster_labels)
df['Cluster Labels'] = cluster_labels  # 将聚类标签添加到DataFrame
df['TF-IDF Labels'] = text_labels  # 将TF-IDF生成的标签添加到DataFrame
df_result = df[['text', 'Cluster Labels', 'TF-IDF Labels']]  # 仅选择需要的列
df_result.to_csv('Texts_with_TFIDF_Labels_WOMEN.csv', index=False)  # 导出为CSV文件
print(cluster_labels)
print("CSV file has been created with texts and their TF-IDF labels.")
