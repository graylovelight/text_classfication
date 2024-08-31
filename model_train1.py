import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# 二分类数据集 —— 模型训练（修改型，花里胡哨的）

vocab_size = 20000
maxlen = 200

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


# 模型建立与训练
inputs = Input(shape=(maxlen,))
use_pretrained_wv = True
if use_pretrained_wv:
    wv = Embedding(vocab_size+1, 300, input_length=maxlen, weights=[embedding_matrix])(inputs)
else:
    wv = Embedding(vocab_size+1, 300, input_length=maxlen)(inputs)
h = Bidirectional(LSTM(128, kernel_regularizer=l2(0.01)))(wv)
y = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(h)

# 创建模型
m = tf.keras.Model(inputs=inputs, outputs=y)
m.summary()

# 编译模型
m.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
m.fit(X_train, y_train, batch_size=64, epochs=12, validation_split=0.15, callbacks=[early_stopping])

# 使用测试集评估模型
evaluation = m.evaluate(X_test, y_test)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

# 指定保存模型的路径，保存模型
model_path = r'训练好的模型\model.h5'
m.save(model_path)
