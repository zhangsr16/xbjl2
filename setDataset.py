import pandas as pd
import shutil
import os

seqlen = 4

def spilt_dataset(union_dfs, dataset_folder, exclude_columns):
    df1 = union_dfs[-1]
    df2 = union_dfs[-2]
    df3 = union_dfs[-3]
    # 创建一个新的DataFrame用于存储差值
    up_df = pd.DataFrame(index=df1.index, columns=df1.columns)
    down_df = pd.DataFrame(index=df1.index, columns=df1.columns)

    # 计算差值，排除指定的列
    for column in df1.columns:
        if column in exclude_columns:
            up_df[column] = df1[column]  # 直接复制不参与差值计算的列
        else:
            up_df[column] = (df1[column] - df2[column]) / df2[column]
            down_df[column] = (df2[column] - df3[column]) / df3[column]

    # 根据指定列的差值将数据分为两份
    if len(union_dfs) == seqlen + 1:
        positive_diff = (up_df['最高'] > 0) & (down_df['最低'] < 0)
        negative_diff = (up_df['最高'] < 0) & (down_df['最低'] > 0)
    else:
        positive_diff = up_df['最高'] >= 0
        negative_diff = down_df['最低'] < 0
    pos = 0
    for df in union_dfs:
        if pos == seqlen:
            break

        Up_dataset = df[positive_diff]
        Other_dataset = df[negative_diff]
        Up_dataset.to_excel(dataset_folder + 'APP_1/Up_' + str(pos) + '.xlsx', index=False)
        Other_dataset.to_excel(dataset_folder + 'APP_0/Other_' + str(pos) + '.xlsx', index=False)
        pos += 1
    return len(Up_dataset), len(Other_dataset)


# THS
exclude_columns = ['代码', '名称', '日期']  # 需要排除的列名列表，根据实际情况修改
data_folder = './THS/'

label_files = [file_name for file_name in os.listdir(data_folder)]
df_list = []
# label_files[-seqlen:]，文件内容仅为xlsx时
for label_file in label_files:
    if label_file.endswith('.xlsx'):
        df = pd.read_excel(data_folder + label_file)
        df_list.append(df)


# auto rows_Train
train_up, train_other = spilt_dataset(df_list[:seqlen + 1], './Training_set/', exclude_columns)
test_up, test_other = spilt_dataset(df_list[seqlen - 1: seqlen * 2 + 1], './Testing_set/', exclude_columns)
predict_up, predict_other = spilt_dataset(df_list[seqlen: seqlen * 2 + 1], './Predict_set/', exclude_columns)

# 生成数据集
print(f"Train: {train_up, train_other}, Test {test_up, test_other}, Predict: {predict_up, predict_other}")

