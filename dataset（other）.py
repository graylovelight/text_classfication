import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import copy
import gensim
from sklearn.model_selection import train_test_split
import pickle

# 读取四分类数据集分词后的文本数据，将其转换为索引形式并分为训练集和测试集

datapath = r'多元情感数据集'
emotion_folders = {
    '喜悦': os.listdir(os.path.join(datapath, '喜悦')),
    '愤怒': os.listdir(os.path.join(datapath, '愤怒')),
    '厌恶': os.listdir(os.path.join(datapath, '厌恶')),
    '低落': os.listdir(os.path.join(datapath, '低落'))
}

# 原始数据集的引入，X_orig保存原始数据，Y_orig保存对应的标签
data_all = []
labels_all = []

for emotion, files in emotion_folders.items():
    for file in files:
        file_path = os.path.join(datapath, emotion, file)
        with open(file_path, encoding='utf-8') as f:
            s = f.read()
            data_all.append(s)
            if emotion == '喜悦':
                labels_all.append(0)
            elif emotion == '愤怒':
                labels_all.append(1)
            elif emotion == '厌恶':
                labels_all.append(2)
            elif emotion == '低落':
                labels_all.append(3)

print(len(data_all))
print(len(labels_all))

X_orig = np.array(data_all)
Y_orig = np.array(labels_all)
print("X_orig:", X_orig.shape)
print("Y_orig:", Y_orig.shape)

# 取频率较高的30000个词建立索引字典，将原始数据转化为索引形式的数据，统一填充得到索引数据pad_X
vocab_size = 30000
maxlen = 300

print("Start fitting the corpus......")
t = Tokenizer(vocab_size)
tik = time.time()
t.fit_on_texts(X_orig)
tok = time.time()
word_index = t.word_index
print('all_vocab_size', len(word_index))
print("Fitting time:", (tok - tik), 's')
print("Start vectorizing the sentences.......")
v_X = t.texts_to_sequences(X_orig)
print("Start padding......")
pad_X = pad_sequences(v_X, maxlen=maxlen, padding='post')
print("Finished!")

# 得到30000个高频词的索引字典small_word_index
x = list(t.word_counts.items())
s = sorted(x, key=lambda p: p[1], reverse=True)
small_word_index = copy.deepcopy(word_index)
print("Removing less freq words from word-index dict...")
for item in s[30000:]:
    small_word_index.pop(item[0])
print("Finished!")
print(len(small_word_index))
print(len(word_index))

# 引入词与词向量的对应字典wv_model，得到索引与词向量的对应字典embedding_matrix
model_file = r'sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5'
print("Loading word2vec model......")
wv_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
embedding_matrix = np.random.uniform(size=(vocab_size + 1, 300))  # +1是要留一个给index=0
print("Transferring to the embedding matrix......")

for word, index in small_word_index.items():
    try:
        word_vector = wv_model[word]
        embedding_matrix[index] = word_vector
    except KeyError:
        print(f"Word: [{word}] not in wvmodel! Use random embedding instead.")
print("Finished!")
print("Embedding matrix shape:\n", embedding_matrix.shape)

# 将索引数据配合其标签分成训练集和测试集
np.random.seed(1)
random_indexs = np.random.permutation(len(pad_X))
X = pad_X[random_indexs]
Y = Y_orig[random_indexs]
print(Y[:50])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("训练集中各标签数量:", np.bincount(y_train))
print("测试集中各标签数量:", np.bincount(y_test))

# 要存储的文件路径
embedding_matrix_path = r'嵌入矩阵\embedding_matrix.pkl'
small_word_index_path = r'索引字典\small_word_index.pkl'

train_path_x = r'特征数据\训练集\X_train.pkl'
train_path_y = r'特征数据\训练集\y_train.pkl'

test_path_x = r'特征数据\测试集\X_test.pkl'
test_path_y = r'特征数据\测试集\y_test.pkl'

# 存储 embedding_matrix 和 small_word_index 到文件
with open(embedding_matrix_path, 'wb') as f:
    pickle.dump(embedding_matrix, f)

with open(small_word_index_path, 'wb') as f:
    pickle.dump(small_word_index, f)

# 存储 训练用数据 到文件
with open(train_path_x, 'wb') as f:
    pickle.dump(X_train, f)

with open(train_path_y, 'wb') as f:
    pickle.dump(y_train, f)

# 存储 测试用数据 到文件
with open(test_path_x, 'wb') as f:
    pickle.dump(X_test, f)

with open(test_path_y, 'wb') as f:
    pickle.dump(y_test, f)

print("数据预处理完成并已保存到文件。")
