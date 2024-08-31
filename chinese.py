import pandas as pd
import os
import jieba

# 读取二分类数据集（存储于CSV文件）的文本数据，为其进行分词处理

# 创建存储数据的文件夹
pos_folder = r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\原始数据\Chinese\pos'
neg_folder = r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\原始数据\Chinese\neg'

if not os.path.exists(pos_folder):
    os.makedirs(pos_folder)
if not os.path.exists(neg_folder):
    os.makedirs(neg_folder)

# 读取CSV文件
df = pd.read_csv(r'C:\Users\1\Desktop\电子课本\电子书\创新实践\数据集\原始数据\Chinese\online_shopping_10_cats.csv')

# 按照条件将数据写入不同的文件夹
for index, row in df.iterrows():
    if pd.isnull(row['review']):  # 检查'review'列是否为空
        continue
    processed_review = " ".join(jieba.cut(row['review']))  # 使用 jieba 分词并以空格分隔
    if row['label'] == 1:
        with open(os.path.join(pos_folder, f'{index}.txt'), 'w', encoding='utf-8') as file:
            file.write(processed_review)
    elif row['label'] == 0:
        with open(os.path.join(neg_folder, f'{index}.txt'), 'w', encoding='utf-8') as file:
            file.write(processed_review)
