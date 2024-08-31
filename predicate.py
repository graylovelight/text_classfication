import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, messagebox
# 二分类模型调用 —— GUI界面
# 四分类模型训练情况不太好，先不设置GUI界面

# 创建一个窗口
root = Tk()
root.title("文本情感分析")
root.geometry("500x400")

# 全局变量
X_predict_orig = None
pad_X_predict = None


def load_small_word_index():
    global small_word_index
    with open(r'索引字典\small_word_index.pkl', 'rb') as f:
        small_word_index = pickle.load(f)


def preprocess_text_files(files):
    global X_predict_orig, pad_X_predict
    predict_all = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            s = f.read()
            processed_review = " ".join(jieba.cut(s))  # 使用 jieba 分词并以空格分隔
            predict_all.append(processed_review)

    X_predict_orig = np.array(predict_all)
    t = Tokenizer()
    t.word_index = small_word_index
    v_X_predict = t.texts_to_sequences(X_predict_orig)
    pad_X_predict = pad_sequences(v_X_predict, maxlen=200, padding='post')


def predict_sentiment():
    global pad_X_predict
    # 指定读取模型的路径
    model_path = r'训练好的模型\model.h5'

    # 加载模型
    loaded_model = tf.keras.models.load_model(model_path)

    # 使用加载的模型进行预测
    predictions = loaded_model.predict(pad_X_predict)

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


def upload_files():
    files = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
    if files:
        preprocess_text_files(files)
        messagebox.showinfo("上传成功", "文件上传成功！可点击按钮‘开始分析’进行预测")
    else:
        messagebox.showwarning("未选择文件", "请上传待预测的文本数据！")


# 创建按钮和标签
upload_button = Button(root, text="上传文件", command=upload_files, padx=20, pady=10)
upload_button.pack(pady=20)

predict_button = Button(root, text="开始分析", command=predict_sentiment, padx=20, pady=10)
predict_button.pack(pady=20)

# 加载词索引字典
load_small_word_index()

# 启动 GUI 主循环
root.mainloop()
