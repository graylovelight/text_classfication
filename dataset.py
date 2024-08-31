import os
import numpy as np
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import copy
import gensim
from sklearn.model_selection import train_test_split
import pickle

# 读取分词后的训练集文本数据，将其转换为特征形式并分为训练集和测试集

datapath = r'数据集\原始数据'
pos_files = os.listdir(datapath+'/pos')
neg_files = os.listdir(datapath+'/neg')

#原始数据集的引入，X_orig保存原始数据，Y_orig保存对应的标签
pos_all = []
neg_all = []
for pf,nf in zip(pos_files,neg_files):
    with open(datapath+'/pos'+'/'+pf,encoding='utf-8') as f:
        s = f.read()
        pos_all.append(s)
    with open(datapath+'/neg'+'/'+nf,encoding='utf-8') as f:
        s = f.read()
        neg_all.append(s)
print(len(pos_all))
print(len(neg_all))
X_orig = np.array(pos_all+neg_all)
Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])
print("X_orig:",X_orig.shape)
print("Y_orig:",Y_orig.shape)

#取频率较高的20000个词建立索引字典，将原始数据转化为索引形式的数据，统一填充得到索引数据pad_X
vocab_size = 20000
maxlen = 200
print("Start fitting the corpus......")
t = Tokenizer(vocab_size) # 要使得文本向量化时省略掉低频词，就要设置这个参数
tik = time.time()
t.fit_on_texts(X_orig) # 在所有的评论数据集上训练，得到统计信息
tok = time.time()
word_index = t.word_index # 不受vocab_size的影响
print('all_vocab_size',len(word_index))
print("Fitting time: ",(tok-tik),'s')
print("Start vectorizing the sentences.......")
v_X = t.texts_to_sequences(X_orig) # 受vocab_size的影响
print("Start padding......")
pad_X = pad_sequences(v_X,maxlen=maxlen,padding='post')
print("Finished!")

#得到20000个高频词的索引字典small_word_index
x = list(t.word_counts.items())# 获取包含词与词频的列表
s = sorted(x,key=lambda p:p[1],reverse=True)# 按词频进行降序排序得到s
small_word_index = copy.deepcopy(word_index)
print("Removing less freq words from word-index dict...")
for item in s[20000:]:
    small_word_index.pop(item[0])# 移除低频词
print("Finished!")
print(len(small_word_index))
print(len(word_index))

#引入词与词向量的对应字典wv_model，得到索引与词向量的对应字典embedding_matrix
model_file = r'GoogleNews-vectors-negative300.bin'
print("Loading word2vec model......")
wv_model = gensim.models.KeyedVectors.load_word2vec_format(model_file,binary=True)
embedding_matrix = np.random.uniform(size=(vocab_size+1,300)) # +1是要留一个给index=0
print("Transfering to the embedding matrix......")

for word,index in small_word_index.items():
    try:
        word_vector = wv_model[word]
        embedding_matrix[index] = word_vector
    except:
        print("Word: [",word,"] not in wvmodel! Use random embedding instead.")
print("Finished!")
print("Embedding matrix shape:\n",embedding_matrix.shape)

#将索引数据配合其标签分成训练集和测试集
np.random.seed = 1
random_indexs = np.random.permutation(len(pad_X))
X = pad_X[random_indexs]
Y = Y_orig[random_indexs]
print(Y[:50])
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_test:",X_test.shape)
print("y_test:",y_test.shape)
print(list(y_train).count(1))
print(list(y_train).count(0))


# 要存储的文件路径
embedding_matrix_path = r'嵌入矩阵\embedding_matrix.pkl'
small_word_index_path = r'索引字典\small_word_index.pkl'

train_path_x = r'数据集\特征数据\训练集\X_train.pkl'
train_path_y = r'数据集\特征数据\训练集\y_train.pkl'

test_path_x = r'数据集\特征数据\测试集\X_test.pkl'
test_path_y = r'数据集\特征数据\测试集\y_test.pkl'

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
with open(test_path_x , 'wb') as f:
    pickle.dump(X_test, f)

with open(test_path_y, 'wb') as f:
    pickle.dump(y_test, f)