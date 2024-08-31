import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import pickle
# 处理原始文本数据为特征形式

# 加载small_word_index
with open(r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\索引字典\small_word_index.pkl', 'rb') as f:
    small_word_index = pickle.load(f)

# 加载原始文本数据
predict_data_path = r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\待预测数据\原始数据'
predict_files = os.listdir(predict_data_path)

# 读取数据并进行分词处理
predict_all = []
for pf in predict_files:
    with open(os.path.join(predict_data_path, pf), encoding='utf-8') as f:
        s = f.read()
        processed_review = " ".join(jieba.cut(s))  # 使用 jieba 分词并以空格分隔
        predict_all.append(processed_review)

print(len(predict_all))
X_predict_orig = np.array(predict_all)
print("X_predict_orig:", X_predict_orig.shape)

# 使用small_word_index将数据转换为特征形式
t = Tokenizer()
t.word_index = small_word_index

print("Start vectorizing the sentences.......")
v_X_predict = t.texts_to_sequences(X_predict_orig)
print("Start padding......")
pad_X_predict = pad_sequences(v_X_predict, maxlen=200, padding='post')
print("Finished!")

# 将处理后的特征数据保存到文件
predict_path_x = r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\待预测数据\特征数据\X_predict.pkl'

with open(predict_path_x, 'wb') as f:
    pickle.dump(pad_X_predict, f)

print("待预测数据处理完成，并已保存到文件:", predict_path_x)
