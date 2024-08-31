import pandas as pd
import os
import jieba
# 读取四分类数据集（存储与CSV文件）的文本数据，为其进行分词处理

# 文件路径
data_file = r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\原始数据\多元情感数据集\simplifyweibo_4_moods.csv'
output_folder = r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\原始数据\多元情感数据集'

# 创建情感类别文件夹
emotion_folders = {
    0: os.path.join(output_folder, '喜悦'),
    1: os.path.join(output_folder, '愤怒'),
    2: os.path.join(output_folder, '厌恶'),
    3: os.path.join(output_folder, '低落')
}

# 确保文件夹存在
for folder in emotion_folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

# 读取CSV文件
df = pd.read_csv(data_file)

# 按照条件将数据写入不同的文件夹
for index, row in df.iterrows():
    if pd.isnull(row['review']):
        continue
    processed_review = " ".join(jieba.cut(row['review']))
    label = row['label']
    if label in emotion_folders:
        file_path = os.path.join(emotion_folders[label], f'{index}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(processed_review)

print("数据处理完成，并已保存到相应文件夹中")
