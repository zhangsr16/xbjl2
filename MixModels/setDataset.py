import pandas as pd
import shutil
import os

def give_rows(df_list, key_column, common_rows, seq_start, seq_end):
    # 遍历所有DataFrame，删除差异行
    updated_dfs = []
    for df in df_list[seq_start: seq_end]:
        df = df[df[key_column].isin(common_rows)]
        updated_df = df.reset_index(drop=True)
        updated_dfs.append(updated_df)
    return updated_dfs


def scale_rows(df_list, key_column, seq_start, seq_end):
    # 进行内连接，找出所有DataFrame中的共同行
    common_rows = df_list[seq_start]
    pos = 0
    for df in df_list[seq_start + 1:seq_end]:
        pos += 1
        common_rows = pd.merge(common_rows, df, on=key_column, how='inner',
                               suffixes=('_left' + str(pos), '_right' + str(pos)))

    # 遍历所有DataFrame，删除差异行
    updated_dfs = []
    for df in df_list[seq_start: seq_end]:
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
            diff_df[column] = (df1[column] - df2[column]) / df2[column]

    # 根据指定列的差值将数据分为两份
    if len(union_dfs) == 5:
        positive_diff = diff_df[column_for_diff] > 0.04
        negative_diff = diff_df[column_for_diff] < -0.04
    else:
        positive_diff = diff_df[column_for_diff] > 0
        negative_diff = diff_df[column_for_diff] < 0
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
        Up_dataset.to_excel(dataset_folder + 'APP_1/Up_' + str(pos) + '.xlsx', index=False)
        Other_dataset.to_excel(dataset_folder + 'APP_0/Other_' + str(pos) + '.xlsx', index=False)
        pos += 1
    return len(Up_dataset), len(Other_dataset)


# THS
column_for_diff = '最高'  # 根据你需要的列名进行修改
exclude_columns = ['代码', '名称', '所属行业', 'DATE']  # 需要排除的列名列表，根据实际情况修改
data_folder = './THS/'
seqlen = 4

label_files = [file_name for file_name in os.listdir(data_folder)]
df_list = []
# label_files[-seqlen:]，文件内容仅为xlsx时
for label_file in label_files:
    if label_file.endswith('.xlsx'):
        df = pd.read_excel(data_folder + label_file)
        df_list.append(df)

# # given rows
# with open('zsr.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
# lines = [lines.strip() for lines in lines]
# union_dfs = give_rows(df_list, '代码', lines)

# auto rows
union_dfs = scale_rows(df_list, '代码', 0, seqlen + 1)  # Train
train_up, train_other = spilt_dataset(union_dfs, './Training_set/', column_for_diff, exclude_columns)
union_dfs = scale_rows(df_list, '代码', seqlen - 1, seqlen * 2 + 1)  # Test
test_up, test_other = spilt_dataset(union_dfs, './Testing_set/', column_for_diff, exclude_columns)
union_dfs = scale_rows(df_list, '代码', seqlen, seqlen * 2 + 1)  # Predict
predict_up, predict_other = spilt_dataset(union_dfs, './Predict_set/', column_for_diff, exclude_columns)

# 生成数据集
print(f"Train: {train_up, train_other}, Test {test_up, test_other}, Predict: {predict_up, predict_other}")


# EnvSet
data_folder = 'C:/Users/z30060762/Desktop/Ptest/THSData-main/Data/Dataset-master/XLSX/'

label_folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
total_dfs = ['Area', 'Field']
# Read all option_dirs
for label_folder in label_folders:
    folder_id = os.path.basename(label_folder)
    if folder_id not in total_dfs:
        continue
    # Read all cycle_dirs
    for f in os.scandir(label_folder):
        cycle_id = os.path.basename(f)
        # Read all layer_CSV
        pos = 0
        for file_name in reversed(os.listdir(f)):
            if pos == seqlen:
                break
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(f, file_name)
                destination_path = './Env_set/' + folder_id + '/' + cycle_id + '/' + file_name
                shutil.copy(file_path, destination_path)
                pos += 1

