import json
import logging
import os
import struct
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import re


def read_config(path):
    """
    read config file based on the giving path.
    :param path:
    """
    with open(path, 'r') as file:
        config_dict = json.load(file)

    return config_dict


def cosine_classsifier(x, w):
    """
    cosine_classsifier
    :param x:
    :param w:
    :return:
    """
    x = F.normalize(x, dim=1)
    w = F.normalize(w, dim=0)

    return torch.matmul(x, w)


def binary_embedding(data, max_len=None):
    """
    binary embedding function.
    :param data:
    :param max_len:
    :return:
    """
    data_new = [[] for _ in range(data.shape[0])]
    data_max = np.max(data, axis=0)

    if max_len is None:
        max_len = []
        for i in range(data.shape[1]):
            # Number of binary bits of the maximum value of the feature.
            max_len.append(len(np.binary_repr(int(data_max[i]) + 1)))

    # Embedding all the features of each sample: concatenate the binary representation of all statistical feature values
    for i in range(data.shape[0]):  # row
        for j in range(data.shape[1]):  # col
            # Initialize an array according to the maximum number of digits
            data_new[i].append(np.zeros(shape=abs(max_len[j]), dtype=np.float32))

            if max_len[j] <= 0:  # do nothing
                continue
            # Binary representation of the current feature of the current sample.
            if data[i][j] < 0:
                print("Negative value:{} is invalid, it will be treated as zero.".format(data[i][j]))
                data[i][j] = 0
            binary_str = np.binary_repr(int(data[i][j]))

            try:
                # Assign values from the least significant bit to the most significant bit.
                for k in range(len(binary_str)):
                    data_new[i][-1][-(k + 1)] = int(binary_str[-(k + 1)])
            except Exception as e:
                print("binary embedding error:", e)
        data_new[i] = np.concatenate(data_new[i])

    return np.stack(data_new)


def get_output_dim(app_config_path):
    """
    :param app_config_path:
    :return:
    """
    app_config = pd.read_csv(app_config_path).dropna(subset=['AppName'])
    return len(app_config)


def get_threshold_from_file(result_path, target_recall):
    """
    Read the threshold of the 'others' corresponding to the given target recall from the JSON file,
    as well as the actual recall corresponding to the threshold
    :param result_path:
    :param target_recall:
    :return:
    """
    with open(result_path, 'r') as file:
        result_json = json.load(file)

    threshold, actual_recall = result_json["target_others_recall_{}_threshold".format(target_recall)]
    return threshold, actual_recall


def convert_payload_to_list(payloads, padding_to):
    """
    :param payloads: Original payload string.
    :param padding_to: Specify the length in bytes. Pad with zeros if insufficient.
    """
    # if it is a valid payload list string, just convert it to list.
    if isinstance(payloads, str) and payloads.startswith("[") and payloads.endswith("]"):
        return eval(payloads)
    # if it is a valid payload list, do nothing.
    if isinstance(payloads, list):
        return payloads
    try:
        if isinstance(payloads, str):
            payloads = payloads.replace(".", "")
            payloads = payloads.split(',')[0]
        elif isinstance(payloads, int):
            payloads = str(payloads)
        
        bytes_list = [int(payloads[i: i + 2], 16) for i in range(0, min(len(payloads), padding_to * 2), 2)]
    except ValueError as _:
        return np.nan

    if len(bytes_list) < padding_to:
        bytes_list.extend([0] * (padding_to - len(bytes_list)))
    return bytes_list


def flow_5_tuple(row):
    """get 5 tuple info for a flow"""
    if row["ip.proto"] == 6:
        return f'{row["ip.src"]}_{row["ip.dst"]}_{row["tcp.srcport"]}_{row["tcp.dstport"]}_{row["ip.proto"]}'
    elif row["ip.proto"] == 17:
        return f'{row["ip.src"]}_{row["ip.dst"]}_{row["udp.srcport"]}_{row["udp.dstport"]}_{row["ip.proto"]}'
    else:
        raise ValueError("invalid  ip.proto:", row["ip.proto"])


def get_app_config_info(app_config):
    """get app and id mapping"""
    appName_to_id = {}
    id_to_appName = {}
    for _, row in app_config.iterrows():
        if row["AppName"] is not None and row["AppName"].upper() != "NULL":
            appName_to_id[row["AppName"]] = int(row["AppId"])
            id_to_appName[int(row["AppId"])] = row["AppName"]

    return appName_to_id, id_to_appName


def get_logger(log_path=None):
    """default logger"""
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger


def log_confusion_matrix(logger, cm, appconf):
    """Logs AppName at the beginning of each row of the confusion matrix with formatted alignment"""
    max_app_name_length = max(appconf["AppName"].apply(len))
    logger.info('Confusion Matrix:')
    for app_name, row in zip(appconf["AppName"], cm):
        # Format each AppName to be left-aligned within max_app_name_length characters
        # and each number in the row to be right-aligned within 6 characters
        formatted_app_name = f'{app_name:<{max_app_name_length}}'
        row_str = ' '.join(f'{num:>6}' for num in row)
        logger.info(f'{formatted_app_name}: {row_str}')


def load_generated_data_for_train(path, device="cuda", num_samples=10000):
    """load_generated_data_for_train"""
    generated_data = np.load(path)
    sample_indices = np.random.choice(generated_data.shape[0], size=num_samples, replace=False)
    samples = generated_data[sample_indices]
    return data_redirect(samples, device)


def data_redirect(dataset, device):
    """data_redirect"""
    processed = []
    for item in dataset:
        part1, part2, part3 = np.split(item, [59, 315])
        part2 = part2.reshape(1, 2, -1)
        part1 = part1.reshape(1, 1, -1)
        part3 = part3.reshape(1, 1, -1)

        processed.append(([torch.Tensor(part1).to(device),
                           torch.Tensor(part2).to(device),
                           torch.Tensor(part3).to(device)],
                          (torch.tensor(0).to(device), AppendixInfo("Generated other", part3.sum()))))

    return processed


def generate_bin_file(output_path, blacklist_input_path, target_protoIds_path, app_config_path, threshold):
    """write the info into the binary file."""
    blacklist_df = pd.read_csv(blacklist_input_path)
    blacklist = {}
    for row_id, row in blacklist_df.iterrows():
        blacklist[row["protoId"]] = row["type"]
    blacklist_len = len(blacklist)
    print("blacklist_len: ", blacklist_len)

    app_config_df = pd.read_csv(app_config_path)
    output_dim = len(app_config_df)
    print("output_dim: ", output_dim)

    target_protoIds_df = pd.read_csv(target_protoIds_path)
    target_proto_num = len(target_protoIds_df)
    print("target_proto_num: ", target_proto_num)

    try:
        binary_stream_dim = struct.pack('B', output_dim)
        binary_stream_threshold = struct.pack('>f', threshold)
        binary_blacklist_len = struct.pack('>H', blacklist_len) ## 2 bytes big end
        binary_target_proto_num = struct.pack('>H', target_proto_num) ## 2 bytes big end

        with open(output_path, "wb") as file:
            file.write(binary_stream_dim)
            file.write(binary_stream_threshold)
            file.write(binary_blacklist_len)

            for protoId, itemType in blacklist.items():
                file.write(struct.pack('>i', protoId))
                file.write(struct.pack('B', itemType))
            file.write(binary_target_proto_num)
          
            for _, row in target_protoIds_df.iterrows():
                protoId = row['protoId']
                file.write(struct.pack('>I', protoId))

    except Exception as error:
        print("error")
        return False

def read_bin_file_with_blacklist(bin_path):
    with open(bin_path, 'rb') as file:
        char_data = file.read(1)
        
        float_data = file.read(4)
        float_value = struct.unpack('>f', float_data)[0]
        
        print(f'output_dim: {char_data[0]}')
        print(f'threshold: {float_value}')
        
        protocol_len = file.read(2)
        protocol_len = struct.unpack('>H', protocol_len)[0]
        print(f'protocol_len: {protocol_len}')
        
        cnt = 0
        while cnt < protocol_len:
            int_data = file.read(4)
            char_data = file.read(1)
            
            int_value = struct.unpack('>i', int_data)[0]
            char_value = char_data[0]
            
            print(int_value)
            print(char_value)
            cnt += 1
            
            
        target_protocol_len = file.read(2)
        target_protocol_len = struct.unpack('>H', target_protocol_len)[0]
        print(f'target_protocol_len: {target_protocol_len}')
        
        cnt = 0
        while cnt < target_protocol_len:
            int_data = file.read(4)
            
            if len(int_data) == 0:
                break
            
            int_value = struct.unpack('>i', int_data)[0]
            
            print(int_value)
            cnt += 1


AppendixInfo = namedtuple('AppendixInfo', ['flow_5tuple', 'flow_bytes'])

def data_split(input_dir, output_dir, test_size=0.2):
    """
    For temporary use in full_dataset splitting, this logic will be migrated to NCE when used commercially.
    Args:
        input_dir (str): Path to the input directory containing subdirectories of CSV files.
        output_dir (str): Path to the output directory where training and testing sets will be saved.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    # check input_dir
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"'{input_dir}' is not a directory.")

    train_dir = os.path.join(output_dir, 'Training_set')
    test_dir = os.path.join(output_dir, 'Testing_set')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # iterate all sub dirs.
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path):
            # merge all csv files of one app into a single dataframe.
            all_data = []
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    df["file_name"] = file_name
                    all_data.append(df)
                elif file_name.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    df["file_name"] = file_name
                    all_data.append(df)
                
            all_data = pd.concat(all_data, ignore_index=True)

            if len(all_data) == 0:
                continue

            # train test split
            train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=42)

            # create output sub dir.
            train_subdir = os.path.join(train_dir, subdir)
            test_subdir = os.path.join(test_dir, subdir)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)

            # save the training and testing set.
            for group_name, group_df in train_data.groupby("file_name"):
                group_df = group_df.drop('file_name', axis=1)
                new_sample_count = len(group_df)
                new_group_name = re.sub(r'-(\d+)-train-', f'-{new_sample_count}-train-', group_name)
                group_df.to_csv(os.path.join(train_subdir, new_group_name), index=False)
            for group_name, group_df in test_data.groupby("file_name"):
                group_df = group_df.drop('file_name', axis=1)
                new_sample_count = len(group_df)
                new_group_name = re.sub(r'-(\d+)-train-', f'-{new_sample_count}-train-', group_name)
                group_df.to_csv(os.path.join(test_subdir, new_group_name), index=False)


if __name__ == '__main__':
    bin_path = '/home/wangmowei/2024/traffic_classification_2024/artifacts/model_package_deploy/model_package-1.2.21/config'
    read_bin_file_with_blacklist(bin_path)
    
    # data_split("../../data/basic/gre/full_dataset", "../../data/basic/gre", test_size=0.2)
