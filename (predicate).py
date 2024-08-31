import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
# 加载模型，导入文本的特征数据，进行文本情感分类

# 指定读取模型的路径
model_path = r'训练好的模型\model.h5'

# 加载模型
loaded_model = tf.keras.models.load_model(model_path)

# 加载待预测数据
predict_path_x = r'待预测数据\特征数据\X_predict.pkl'

with open(predict_path_x, 'rb') as f:
    X_predict = pickle.load(f)

# 使用加载的模型进行预测
predictions = loaded_model.predict(X_predict)

# 将预测结果转化为标签
predicted_labels = (predictions > 0.5).astype(int).flatten()

# 输出预测结果
for i, prediction in enumerate(predicted_labels):
    print(f"Review {i + 1}: {'Positive' if prediction == 1 else 'Negative'}")

# 统计正面和负面预测的数量
positive_count = np.sum(predicted_labels)
negative_count = len(predicted_labels) - positive_count

# 绘制条形图
labels = ['Positive', 'Negative']
counts = [positive_count, negative_count]

plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color=['green', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Analysis of Reviews')
plt.show()
