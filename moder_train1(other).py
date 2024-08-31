import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
import pickle
from collections import Counter
# 四分类数据集 —— 模型训练

# 参数设置
vocab_size = 30000
maxlen = 300
embedding_dim = 300

# 文件路径
embedding_matrix_path = r'嵌入矩阵\embedding_matrix.pkl'
train_path_x = r'特征数据\训练集\X_train.pkl'
train_path_y = r'特征数据\训练集\y_train.pkl'
test_path_x = r'特征数据\测试集\X_test.pkl'
test_path_y = r'特征数据\测试集\y_test.pkl'

# 检查数据文件是否存在，如果存在则加载数据，否则执行数据处理代码
if os.path.exists(embedding_matrix_path) and os.path.exists(train_path_x) \
    and os.path.exists(train_path_y) and os.path.exists(test_path_x) \
    and os.path.exists(test_path_y):
    # 加载 embedding_matrix
    with open(embedding_matrix_path, 'rb') as f:
        embedding_matrix = pickle.load(f)

    # 加载 X_train 和 y_train
    with open(train_path_x, 'rb') as f:
        X_train = pickle.load(f)

    with open(train_path_y, 'rb') as f:
        y_train = pickle.load(f)

    # 加载 X_test 和 y_test
    with open(test_path_x, 'rb') as f:
        X_test = pickle.load(f)

    with open(test_path_y, 'rb') as f:
        y_test = pickle.load(f)
else:
    print("所需数据不足，请运行dataset程序.")
    exit()

# 使用部分训练数据
subset_size = 10000  # 使用前10000条数据进行快速训练
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]

# 检查标签分布
print("Train labels distribution (subset):", Counter(y_train_subset))
print("Test labels distribution:", Counter(y_test))

# 将标签转化为独热编码
num_classes = 4
y_train_subset = to_categorical(y_train_subset, num_classes)
y_test = to_categorical(y_test, num_classes)

# 模型建立与训练
model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=True))
model.add(Bidirectional(LSTM(64)))  # 使用一个双向LSTM层
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 调整学习率
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 10))

# 训练模型
history = model.fit(X_train_subset, y_train_subset, batch_size=32, epochs=10, validation_split=0.15, callbacks=[lr_scheduler])

# 使用测试集评估模型
evaluation = model.evaluate(X_test, y_test)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

# 指定保存模型的路径，保存模型
model_path = r'训练好的模型\model.h5'
model.save(model_path)
