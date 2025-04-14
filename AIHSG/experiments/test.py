from sentence_transformers import SentenceTransformer, util

# 加载预训练模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 输入文本
text1 = "Hello, how are you?"
text2 = "Hi there! How's it going?"

# 转换文本为语义向量
vector1 = model.encode(text1, convert_to_tensor=True)
vector2 = model.encode(text2, convert_to_tensor=True)

# 计算余弦相似度
cosine_similarity = util.pytorch_cos_sim(vector1, vector2)

# 打印相似度
print(f"Cosine Similarity: {cosine_similarity.item()}")
