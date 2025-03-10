
import pandas as pd
import numpy as np
import os


def scale_rows(df_list, key_column):
    # 进行内连接，找出所有DataFrame中的共同行
    common_rows = df_list[0]
    pos = 0
    for df in df_list[1:]:
        pos += 1
        common_rows = pd.merge(common_rows, df, on=key_column, how='inner',
                               suffixes=('_left' + str(pos), '_right' + str(pos)))

    # 遍历所有DataFrame，删除差异行
    updated_dfs = []
    for df in df_list:
        df_diff = df[~df[key_column].isin(common_rows[key_column])]
        df = df.drop(df_diff.index)
        updated_df = df.reset_index(drop=True)
        updated_dfs.append(updated_df)

    return updated_dfs


def spilt_dataset(union_dfs, dataset_folder, column_for_diff, exclude_columns):
    df1 = union_dfs[-1]
    df2 = union_dfs[-2]
    # 创建一个新的DataFrame用于存储差值
    diff_df = pd.DataFrame(index=df1.index, columns=df1.columns)

    # 计算差值，排除指定的列
    for column in df1.columns:
        if column in exclude_columns:
            diff_df[column] = df1[column]  # 直接复制不参与差值计算的列
        else:
            diff_df[column] = df1[column] - df2[column]

    # 根据指定列的差值将数据分为两份
    positive_diff = diff_df[column_for_diff] > 0
    negative_diff = diff_df[column_for_diff] <= 0
    pos = 0
    for df in union_dfs:
        # 列交换
        df.insert(0, '所属行业_temp', df['所属行业'])
        df.insert(0, 'DATE_temp', df['DATE'])
        df = df.drop('所属行业', axis=1)
        df = df.drop('DATE', axis=1)
        df = df.rename(columns={'所属行业_temp': '所属行业'})
        df = df.rename(columns={'DATE_temp': 'DATE'})

        Up_dataset = df[positive_diff]
        Other_dataset = df[negative_diff]
        Up_dataset.to_excel(dataset_folder + 'APP_0/Other_' + str(pos) + '.xlsx', index=False)
        Other_dataset.to_excel(dataset_folder + 'APP_1/Up_' + str(pos) + '.xlsx', index=False)
        pos += 1
    return len(Up_dataset)

# # Area
# column_for_diff = '占市场'  # 根据你需要的列名进行修改
# exclude_columns = ['序号', '地区']  # 需要排除的列名列表，根据实际情况修改
# THS
column_for_diff = '最高'  # 根据你需要的列名进行修改
exclude_columns = ['代码', '名称', '所属行业', 'DATE']  # 需要排除的列名列表，根据实际情况修改

data_folder = 'F:/Desktop/THS/data/test2/'
dataset_folder = './Training_set/'
seqlen = 8

label_files = [file_name for file_name in os.listdir(data_folder)]
df_list = []
# label_files[-seqlen:]，文件内容仅为xlsx时
for label_file in label_files:
    if label_file.endswith('.xlsx'):
        df = pd.read_excel(data_folder + label_file)
        df_list.append(df)
union_dfs = scale_rows(df_list, '代码')

# 生成数据集
positive_cnt = spilt_dataset(union_dfs, dataset_folder, column_for_diff, exclude_columns)
