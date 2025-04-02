import hashlib
import numbers
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 将 4 替换为您希望使用的核心数
import random
import sys
from itertools import chain
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import Isomap
from metric_learn import LMNN
from hmmlearn import hmm
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score

from utils import read_config, get_logger


def centerETF(ETF_tensor):
    data_tensor_total = torch.nan_to_num(ETF_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    epsilon = 1e-6
    data_tensor = data_tensor_total[..., -1]
    seq_rate = (data_tensor[..., 1:] - data_tensor[..., :-1]) / (data_tensor[..., :-1] + epsilon)  # 末位维序列
    seq_mean = seq_rate.mean(dim=-1, keepdim=True)  # 最后一维求均
    seq_scale = (seq_rate - seq_mean) / (seq_mean + epsilon)  # 变化率中心化
    # 将三维张量转换为二维数组
    total_reshaped = data_tensor_total.view(data_tensor_total.shape[0], -1)
    seq_reshaped = seq_scale.view(seq_scale.shape[0], -1)

    # 使用K-Means聚类算法进行聚类分析
    # 肘部法
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init=k, random_state=0)
        kmeans.fit(seq_reshaped)
        sse.append(kmeans.inertia_)
    # 轮廓系数
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, n_init=k, random_state=0)
        labels = kmeans.fit_predict(seq_reshaped)
        silhouette_scores.append(silhouette_score(seq_reshaped, labels))

    # 输出最佳聚类数
    best_k_elbow = np.argmax(np.diff(sse)) + 1
    best_k_silhouette = np.argmax(silhouette_scores) + 2
    best_k = max(best_k_elbow, best_k_silhouette)

    kmeans = KMeans(n_clusters=best_k, n_init=best_k, max_iter=best_k, random_state=0)
    kmeans.fit(seq_reshaped)
    # 获取聚类中心
    centroids = kmeans.cluster_centers_
    centroids_tensor = torch.tensor(centroids).float()
    # 计算每个点到聚类中心的Euclidean距离
    distances = torch.cdist(seq_reshaped.unsqueeze(1), centroids_tensor.unsqueeze(0)).squeeze(1)

    # 设置sigma参数，衰减率正比于中心数
    sigma = 1 / best_k
    # 计算权重矩阵，使用高斯核函数
    weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))
    # 归一化权重矩阵，使每列的权重和为1
    weights /= weights.sum(axis=0, keepdims=True)
    # 计算加权均值，得到新的聚类中心
    new_cluster_centers = np.dot(weights.T, total_reshaped)
    centroids_tensor = torch.tensor(new_cluster_centers).float()
    ResETF = centroids_tensor.view(best_k, data_tensor_total.shape[1], data_tensor_total.shape[2], -1)
    return ResETF


def clusterEnv(data_tensor, env_tensors, config):
    data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    epsilon = 1e-6
    seq_rate = (data_tensor[..., 1:] - data_tensor[..., :-1]) / (data_tensor[..., :-1] + epsilon)  # 末位维序列
    seq_mean = seq_rate.mean(dim=-1, keepdim=True)  # 最后一维求均
    TotalCented = (seq_rate - seq_mean) / (seq_mean + epsilon)  # 变化率中心化

    total_tensors = []
    for option in ['Area', 'ETF', 'Field', 'Summary', 'Type']:
        env_tensor = env_tensors[option]
        env_tensor = torch.nan_to_num(env_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        SeqCented = TotalCented[:, config[option + 'Cluster'], :]  # select cols in each cycle, by config define
        # 将三维张量转换为二维数组
        seq_reshaped = SeqCented.view(SeqCented.shape[0], -1)

        # 环境变化率张量: batch, feature, seq, cycle
        env_rate = (env_tensor[:, :, 1:, :] - env_tensor[:, :, :-1, :]) / (env_tensor[:, :, :-1, :] + epsilon)
        env_mean = env_rate.mean(dim=2, keepdim=True)  # 第3维求均
        EnvTotalCented = (env_rate - env_mean) / (env_mean + epsilon)
        total_dfs = []
        for cycle in range(env_tensor.shape[-1]):
            # 映射中心
            EnvCented = EnvTotalCented[..., cycle]  # select cols in each cycle, by config define
            env_reshaped = EnvCented[:, config[option + 'EnvCluster'], :]  # select cols in each cycle, by config define
            initial_centers = env_reshaped.view(env_reshaped.shape[0], -1)
            env_total_reshaped = EnvCented.view(EnvCented.shape[0], -1)

            # 使用K-Means聚类算法进行聚类分析，并指定初始中心点
            kmeans = KMeans(n_clusters=initial_centers.shape[0], init=initial_centers, n_init=1,
                            max_iter=initial_centers.shape[0],
                            random_state=0)
            kmeans.fit(seq_reshaped)
            # 获取聚类中心
            centroids = kmeans.cluster_centers_
            centroids_tensor = torch.tensor(centroids).float()
            # 计算每个点到聚类中心的距离
            distances = torch.cdist(seq_reshaped.unsqueeze(1), centroids_tensor.unsqueeze(0)).squeeze(1)
            distances_init = torch.cdist(seq_reshaped.unsqueeze(1), initial_centers.unsqueeze(0)).squeeze(1)

            # 设置sigma参数，衰减率正比于中心数
            sigma = 1 / initial_centers.shape[0]
            # 计算权重矩阵，使用高斯核函数
            weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))
            weights_init = torch.exp(-distances_init ** 2 / (2 * sigma ** 2))
            # 将权重矩阵与初始聚类中心矩阵作矩阵乘法
            weighted_centers = torch.matmul(weights, centroids_tensor.float())  # 自学习中心
            weighted_centers_init = torch.matmul(weights_init, env_total_reshaped.float())  # 原中心
            # 将加权后的聚类中心矩阵转化为原始张量形状
            SeqEnv = weighted_centers.view(SeqCented.shape[0], env_reshaped.shape[1], -1)
            SeqEnvInit = weighted_centers_init.view(SeqCented.shape[0], EnvCented.shape[1], -1)
            # 张量合并batch, feature(cluster_dim), seq
            SeqCluster = torch.concat((SeqEnv, SeqEnvInit), axis=1)
            total_dfs.append(SeqCluster)

        # 张量合并batch, feature(cluster_dim), seq, cycle
        SeqClusters = torch.stack(tuple(total_dfs), dim=-1)
        # 维度重排为 [batch, feature(cluster_dim), cycle, seq]
        SeqClusters = SeqClusters.permute(0, 1, 3, 2)
        # [feature, cycle] 融合为feature
        result = SeqClusters.reshape(SeqClusters.shape[0], SeqClusters.shape[1] * SeqClusters.shape[2], -1)
        total_tensors.append(result)
    total_tensor = torch.concat(tuple(total_tensors), axis=1)
    return total_tensor

class Saudi(Dataset):
    """
    dataset class
    """

    def __init__(self, data, data_tensor, env_tensor, device, config, id_to_appName=None, return_appendix_infos=False):
        """
        # mixed_features(1 x 59)：1.src/dst ports 32;proto 1;tcp.flags 8;ip.ttl 8;ip.hdr_len 4;tcp.hdr_len 6.
        # ip_direction_pkt_len_seqs(2 x 256): ip.directions 8 x 16repeat = 128;ip.pkt_lens 8 x 16 = 128.total 2 x 128.
        # payloads(1 x 256): normalized 256 bytes.
        :param data:
        :param device:
        :param config:
        :param id_to_appName:
        :param return_appendix_infos:
        """
        self.env = clusterEnv(data_tensor, env_tensor, config)
        self.data = data

        self.tensor = data_tensor
        self.device = device
        self.config = config
        self.id2app = id_to_appName
        self.return_appendix_infos = return_appendix_infos

        # preprocess
        self.data.replace([np.inf, -np.inf], 1e-6, inplace=True)
        self.env = torch.nan_to_num(self.env, nan=0.0, posinf=0.0, neginf=0.0)
        self.tensor = torch.nan_to_num(self.tensor, nan=0.0, posinf=0.0, neginf=0.0)
        # change the hdr_len to the actual value in the ip.header

        # collect appendix infos
        if return_appendix_infos:
            pass

        self.labels = self.data['app_id'].values
        tensor = self.tensor[..., 1:]

        # Kernel Vectors
        tensors_df = []
        # Conv: batch_size, in_channels, sequence...
        conv1d = nn.Conv1d(in_channels=self.tensor.shape[1], out_channels=self.tensor.shape[1], kernel_size=2, stride=1)
        conv1d_env = nn.Conv1d(in_channels=self.env.shape[1], out_channels=self.env.shape[1], kernel_size=2, stride=1)
        max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1)
        avg_pool1d = nn.AvgPool1d(kernel_size=2, stride=1)
        tensor_conv = conv1d(self.tensor)
        tensor_conv_env = conv1d_env(self.env)
        tensor_maxpool = max_pool1d(self.tensor)
        tensor_maxpool_env = max_pool1d(self.env)
        tensor_avgpool = avg_pool1d(self.tensor)
        tensor_avgpool_env = avg_pool1d(self.env)
        tensors_df.append(tensor_conv)
        tensors_df.append(tensor_avgpool)
        tensors_df.append(tensor_maxpool)
        tensors_df.append(self.env)
        # tensors_df.append(tensor_conv_env)
        # tensors_df.append(tensor_avgpool_env)
        # tensors_df.append(tensor_maxpool_env)
        # RNN: batch_size, sequence, in_channels
        lstm = nn.LSTM(input_size=tensor.shape[1], hidden_size=tensor.shape[1],
                       num_layers=tensor.shape[-1], batch_first=True)
        lstm_env = nn.LSTM(input_size=self.env.shape[1], hidden_size=self.env.shape[1],
                           num_layers=self.env.shape[-1], batch_first=True)
        h0 = tensor.permute(2, 0, 1)
        c0 = torch.zeros(tensor.shape[-1], tensor.shape[0], tensor.shape[1])
        output_tensor, (hn, cn) = lstm(tensor.permute(0, 2, 1), (h0, c0))
        tensor_RNN = output_tensor.permute(0, 2, 1)
        tensor_RNN_hn = hn.permute(1, 2, 0)
        tensor_RNN_cn = cn.permute(1, 2, 0)
        h0 = self.env.permute(2, 0, 1)
        c0 = torch.zeros(self.env.shape[-1], self.env.shape[0], self.env.shape[1])
        output_tensor, (hn, cn) = lstm_env(self.env.permute(0, 2, 1), (h0, c0))
        tensor_RNNenv = output_tensor.permute(0, 2, 1)
        tensor_RNNenv_hn = hn.permute(1, 2, 0)
        tensor_RNNenv_cn = cn.permute(1, 2, 0)
        tensors_df.append(tensor_RNN)
        tensors_df.append(tensor_RNN_hn)
        tensors_df.append(tensor_RNN_cn)
        tensors_df.append(tensor_RNNenv)
        tensors_df.append(tensor_RNNenv_hn)
        tensors_df.append(tensor_RNNenv_cn)
        # Attention: batch_size, sequence, in_channels
        attention = nn.MultiheadAttention(embed_dim=tensor.shape[1], num_heads=tensor.shape[1],
                                          batch_first=True)
        attention_env = nn.MultiheadAttention(embed_dim=self.env.shape[1], num_heads=self.env.shape[1],
                                              batch_first=True)
        output_tensor, attn_output_weights = attention(tensor.permute(0, 2, 1), tensor.permute(0, 2, 1),
                                                       tensor.permute(0, 2, 1))
        tensor_TF = output_tensor.permute(0, 2, 1)
        tensor_TF_w = attn_output_weights.permute(0, 2, 1)
        output_tensor, attn_output_weights = attention_env(self.env.permute(0, 2, 1), self.env.permute(0, 2, 1),
                                                           self.env.permute(0, 2, 1))
        tensor_TFenv = output_tensor.permute(0, 2, 1)
        tensor_TFenv_w = attn_output_weights.permute(0, 2, 1)
        tensors_df.append(tensor_TF)
        tensors_df.append(tensor_TF_w)
        tensors_df.append(tensor_TFenv)
        tensors_df.append(tensor_TFenv_w)

        # Tensors_con
        tensors = torch.concat(tuple(tensors_df), dim=1)
        # Norm: batch_size, in_channels, sequences
        norm1d = nn.BatchNorm1d(num_features=tensors.shape[1])
        tensors_normed = norm1d(tensors)

        # Classifiers
        classifiers = []
        self.classifyModels = {}
        self.X_train = tensors_normed.view(tensors_normed.shape[0], -1).detach().numpy()
        self.y_train = np.array([self.labels])[0]
        # SVM
        base_svc = SVC(probability=True, kernel='rbf', gamma='scale', random_state=0)
        SVM_Classifier = SelfTrainingClassifier(base_svc, criterion='k_best',
                                                k_best=math.ceil(0.1 * tensors_normed.shape[0]))
        SVM_Classifier.fit(self.X_train, self.y_train)
        self.classifyModels['SVM_Classifier'] = SVM_Classifier
        # DecisionTree
        DecisionTree_Classifier = DecisionTreeClassifier(criterion='gini', max_depth=tensors_normed.shape[-1],
                                                         random_state=0)
        DecisionTree_Classifier.fit(self.X_train, self.y_train)
        self.classifyModels['DecisionTree_Classifier'] = DecisionTree_Classifier
        # NB
        NB_Classifier = GaussianNB()
        NB_Classifier.fit(self.X_train, self.y_train)
        self.classifyModels['NB_Classifier'] = NB_Classifier
        # EM_GaussianMixture
        EM_Classifier = GaussianMixture(n_components=2, random_state=0)
        EM_Classifier.fit(self.X_train)
        self.classifyModels['EM_Classifier'] = EM_Classifier
        # AdaBoost
        AdaBoost_Classifier = AdaBoostClassifier(n_estimators=tensors_normed.shape[-1], random_state=0)
        AdaBoost_Classifier.fit(self.X_train, self.y_train)
        self.classifyModels['AdaBoost_Classifier'] = AdaBoost_Classifier
        # Bagged Decision Trees
        BagTrees_Classifier = BaggingClassifier(estimator=DecisionTree_Classifier,
                                                n_estimators=tensors_normed.shape[-1],
                                                random_state=0)
        BagTrees_Classifier.fit(self.X_train, self.y_train)
        self.classifyModels['BagTrees_Classifier'] = BagTrees_Classifier
        # RF
        RF_Classifier = RandomForestClassifier(n_estimators=tensors_normed.shape[-1], random_state=0)
        RF_Classifier.fit(self.X_train, self.y_train)
        self.classifyModels['RF_Classifier'] = RF_Classifier
        # kNN
        kNN_Classifier = KNeighborsClassifier(n_neighbors=2)
        kNN_Classifier.fit(self.X_train, self.y_train)
        self.classifyModels['kNN_Classifier'] = kNN_Classifier
        # GaussianHMM
        GaussianHMM = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=tensors_normed.shape[-1],
                                      random_state=0)
        GaussianHMM.fit(self.X_train)
        self.classifyModels['GaussianHMM'] = GaussianHMM

        # classifiers_cm
        self.classifiers_tensor, self.classifiers_RPscore = self.classify(self.X_train, self.y_train)
        self.classifiers_tensor_test = None
        self.classifiers_RPscore_test = None

    def classify(self, X_train, y_train):
        classifiers = []
        # SVM
        y_pred = self.classifyModels['SVM_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # DecisionTree
        y_pred = self.classifyModels['DecisionTree_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # NB
        y_pred = self.classifyModels['NB_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # EM_GaussianMixture
        y_pred = self.classifyModels['EM_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # AdaBoost
        y_pred = self.classifyModels['AdaBoost_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # Bagged Decision Trees
        y_pred = self.classifyModels['BagTrees_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # RF
        y_pred = self.classifyModels['RF_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # kNN
        y_pred = self.classifyModels['kNN_Classifier'].predict(X_train)
        classifiers.append(y_pred)
        # GaussianHMM
        y_pred = self.classifyModels['GaussianHMM'].predict(X_train)
        classifiers.append(y_pred)
        classifiers_tensor = torch.stack(tuple(torch.tensor(classifiers)), dim=-1)
        classifiers_RPscore = []
        for classifier in classifiers:
            cm = confusion_matrix(y_train, classifier)
            print(accuracy_score(y_train, classifier), '\n', cm)
            if cm[0, 1] == 0:
                precision = cm[1, 1] / 1.0
            else:
                precision = cm[1, 1] / cm[0, 1]
            if cm[1, 0] == 0:
                recall = cm[1, 1] / 1.0
            else:
                recall = cm[1, 1] / cm[1, 0]
            classifiers_RPscore.append([precision, recall])
        return classifiers_tensor, classifiers_RPscore

    def __len__(self):
        # 确保返回数据的长度
        return len(self.data)

    def __getitem__(self, index):
        # 检查索引是否超出范围
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range")
        x = [self.tensor[index], self.env[index], self.classifiers_tensor[index], self.classifiers_RPscore_test]
        if self.return_appendix_infos:
            y = torch.tensor(self.labels[index], dtype=torch.long).to(self.device), self.appendix_infos[index]
        else:
            y = torch.tensor(self.labels[index], dtype=torch.long).to(self.device)
        return x, y



def read_env(config, data_folder):
    label_folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
    total_dfs = {}
    # Read all option_dirs
    for label_folder in label_folders:
        folder_id = os.path.basename(label_folder)
        tensors_dfs = []
        # Read all cycle_dirs
        for f in os.scandir(label_folder):
            tensor_dfs = []
            # Read all layer_CSV
            for file_name in os.listdir(f):
                if file_name.endswith('.xlsx'):
                    file_path = os.path.join(f, file_name)
                    df = pd.read_excel(file_path)
                    # shape=batch, chanel
                    tensor = torch.tensor(
                        (df[df.columns[config[folder_id]:]].to_numpy()),
                        dtype=torch.float)
                # shape=batch, chanel, seq
                tensor = torch.unsqueeze(tensor, dim=2)
                tensor_dfs.append(tensor)
            tensors = torch.concat(tuple(tensor_dfs), axis=2)
            tensors = torch.unsqueeze(tensors, dim=3)
            tensors_dfs.append(tensors)
        tensors = torch.concat(tuple(tensors_dfs), axis=3)
        if folder_id == 'ETF':
            tensors = centerETF(tensors)
        total_dfs[folder_id] = tensors
    return total_dfs


def read_data(config, data_folder, id_to_label, folder_id_to_label_id=None):
    """
    Read the dataset from the specified directory.
    :param id_to_label: appName to id dict.
    :param folder_id_to_label_id: The folder label id may not be the label id, if that happened we need a mapping dict.
    :param data_folder:
    """
    label_folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    total_dfs = []
    tensors_dfs = []
    for label_folder in label_folders:
        folder_id = int(os.path.basename(label_folder).split("_")[-1])
        label_id = folder_id_to_label_id[folder_id] if folder_id_to_label_id else folder_id
        try:
            app_name = id_to_label[label_id]
        except KeyError as _:
            print("Warning!: Skip the folder id:{}, it doesn't exists in the appConfig.csv.".format(folder_id))
            continue

        # Read all CSV files in the label folder
        pos = 0
        tensor_dfs = []
        for file_name in os.listdir(label_folder):
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(label_folder, file_name)
                df = pd.read_excel(file_path)
                # shape=batch, chanel
                tensor = torch.tensor(
                    (df[df.columns[config['arch_hidden_range']:]].to_numpy()),
                    dtype=torch.float)
                # shape=batch, chanel, seq
                tensor = torch.unsqueeze(tensor, dim=2)
                tensor_dfs.append(tensor)
                if pos == 0:
                    df["app_name"] = app_name
                    df["app_id"] = label_id
                    total_dfs.append(df)
                    pos += 1
            else:
                print("Warning!: no data found in: {}, app_name: {}".format(label_folder, app_name))
        tensors = torch.concat(tuple(tensor_dfs), axis=2)
        tensors_dfs.append(tensors)
    tensor_data = torch.concat(tuple(tensors_dfs), axis=0)
    new_tensors = []
    for i in range(tensor_data.shape[-1]):
        new_tensor = tensor_data[:, config["FieldCluster"][:3], i]
        column_sums = new_tensor.sum(dim=0, keepdim=False)  # 沿第0维求和
        new_tensor = new_tensor / column_sums  # 张量除法支持广播机制（broadcasting），维度自动扩展
        new_tensors.append(new_tensor)
    new_tensors = torch.stack(tuple(new_tensors), dim=-1)
    tensor_data = torch.concat((tensor_data, new_tensors), axis=1)
    total_data = pd.concat(total_dfs).reset_index(drop=True)
    return total_data, tensor_data


def get_data(train_data, test_data, train_tensor, test_tensor, env_data, device, config, id_to_appName):
    """
    get the training and test dataset.
    :param train_data: train DataFrame
    :param test_data: test DataFrame
    :param device:
    :param config:
    :param id_to_appName:
    """
    train_saudi = Saudi(train_data, train_tensor, env_data, device, config, id_to_appName)
    test_saudi = Saudi(test_data, test_tensor, env_data, device, config, id_to_appName)
    train_saudi.classifiers_tensor_test, train_saudi.classifiers_RPscore_test = train_saudi.classify(test_saudi.X_train, test_saudi.y_train)
    test_saudi.classifiers_tensor_test, test_saudi.classifiers_RPscore_test = test_saudi.classify(train_saudi.X_train, train_saudi.y_train)


    total_data_size = sys.getsizeof(train_saudi) * len(train_saudi) + sys.getsizeof(test_saudi) * len(test_saudi)
    print(f"Loaded dataset memory usage: {total_data_size / 1024 ** 2} MB, "
          f"total items: {len(train_data) + len(test_data)}")

    values1, counts1 = np.unique(train_saudi.labels, return_counts=True)
    values2, counts2 = np.unique(test_saudi.labels, return_counts=True)
    if isinstance(config["trainer_samplesize"], str) and config["trainer_samplesize"] == "label_max_count":
        label_max_count = int(max(c for c in counts1))
    elif config["trainer_samplesize"] > 0:
        label_max_count = config["trainer_samplesize"]
    else:
        raise ValueError("Invalid sample size")
    print("Sample size for each class is", label_max_count)

    for li, c in zip(values1, counts1):
        if li in train_saudi.id2app:
            print("train samples for label{} {} is {}".format(li, train_saudi.id2app[li], c))
    print(f"Total original train set size:{len(train_saudi)}")
    for li, c in zip(values2, counts2):
        if li in test_saudi.id2app:
            print("test samples for label{} {} is {}".format(li, test_saudi.id2app[li], c))
    print(f"Total test set size:{len(test_saudi)}")

    testset = [i for i in test_saudi]
    # train data balance.
    balanced_trainset = []
    label_num = []
    for i in sorted(train_saudi.id2app.keys()):
        ids = np.where(train_saudi.labels == i)[0]  # indexes for the label
        if len(ids) == 0:
            print("Warning: training Label{} {} has no data.".format(i, train_saudi.id2app[i]))
            continue
        n_train = len(ids)
        label_num.append(n_train)

        random_index = np.random.permutation(ids)  # shuffle
        n_resample = label_max_count
        if len(random_index) <= n_resample:
            # label balance for training.
            train_index = np.concatenate([random_index, np.random.choice(random_index, n_resample - len(random_index))])
        else:
            train_index = np.random.choice(random_index, n_resample, replace=False)
        balanced_trainset.extend([train_saudi[index] for index in train_index])

    # replace NaN with zeros for train_set and test_set.
    for sample in chain(balanced_trainset, testset):
        for item in sample[0]:
            if torch.is_tensor(item):
                item[torch.isnan(item)] = 1e-6
    print(f"Total balanced train set size:{len(balanced_trainset)}")
    return balanced_trainset, testset


def sum_hashes(digests, f):
    """hash"""
    bitarray = np.unpackbits(np.frombuffer(b''.join(digests), dtype='>B'))
    rows = np.reshape(bitarray, (-1, f))
    return np.sum(rows, 0)


class DataFilter:
    """using for data filter"""

    def __init__(self, logger=None, bits=128):
        self.logger = get_logger() if logger is None else logger
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = read_config(os.path.join(script_dir, 'model_config.json'))
        self.f = bits
        self.f_bytes = self.f // 8
        self.large_weight_cutoff = 50
        self.batch_size = 200
        self.truncate_mask = 2 ** self.f - 1
        self.window = self.config["simhash_window_size"]

    def run(self, input_path, output_path):
        label_folders = [f.path for f in os.scandir(input_path) if f.is_dir()]
        for label_folder in label_folders:
            # Read all CSV files in the label folder
            folder_id = int(os.path.basename(label_folder).split("_")[-1])
            dfs = []
            for file_name in os.listdir(label_folder):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(label_folder, file_name)
                    df = pd.read_csv(file_path)
                    df["file_name"] = file_name
                    dfs.append(df)

            total_df = pd.concat(dfs).reset_index(drop=True)
            if len(total_df) <= self.config["simhash_min_samplesize"]:
                self.logger.info(f"skip simhash procedure for {label_folder}, total data size:{len(total_df)}")
                continue

            total_df["app_id"] = folder_id
            dataset = Saudi(total_df, "cpu", self.config)

            # Data processing within the class.
            simhash_indexes = self.simhash_process(dataset, [folder_id], self.config["simhash_del_times"],
                                                   self.config["simhash_class_min_dist"],
                                                   self.config["simhash_nearby_min_dist"],
                                                   self.config["simhash_over_filter_threshold"])
            simhash_total_df = total_df.iloc[simhash_indexes]
            self.logger.info(f"simhash procedure for {label_folder} is done, original data size:{len(total_df)}, "
                             f"simhashed data size:{len(simhash_total_df)}")

            output_folder_path = os.path.join(output_path, os.path.basename(label_folder))
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            for group_name, group_df in simhash_total_df.groupby("file_name"):
                group_df = group_df.drop('file_name', axis=1)
                group_df.to_csv(os.path.join(output_folder_path, group_name), index=False)

    def get_features_hash(self, data_set):
        """SimHash calculation for Saudi dataset"""
        hashes = []
        labels = []
        flow_5_tuples = []
        for item in tqdm(data_set):
            a = torch.flatten(item[0][0]).tolist()
            b = torch.flatten(item[0][1]).tolist()
            c = torch.flatten(item[0][2]).tolist()

            str_a = []
            str_b = []
            str_c = []

            for j in range(len(a) - self.window + 1):
                tmp = ""
                for k in range(self.window):
                    tmp += str(int(a[j + k]))
                str_a.append(tmp)
            for j in range(len(b) - self.window + 1):
                tmp = ""
                for k in range(self.window):
                    tmp += str(int(b[j + k]))
                str_b.append(tmp)
            for j in range(len(c) - self.window + 1):
                tmp = ""
                for k in range(self.window):
                    tmp += str(int(c[j + k]))
                str_c.append(tmp)
            str_item_feature = str_a + str_b + str_c

            val = self.build_by_features(str_item_feature)
            hashes.append(val)
            labels.append(int(item[1][0]))
            flow_5_tuples.append(item[1][1].flow_5tuple)

        labels = np.array(labels)
        hashes = np.array(hashes)
        return hashes, labels, flow_5_tuples

    def simhash_process(self, data_set, unique_labels, nearby_del_times=1, class_min_dist=999, nearby_min_dist=0,
                        over_filter_threshold=0.5):
        """simhash process """
        hashes, labels, _ = self.get_features_hash(data_set)

        # 1st filter based on nearby data
        all_index = set(range(len(data_set)))
        for i in range(nearby_del_times):
            del_index = set()
            all_index_list = list(all_index)
            for j in range(len(all_index_list) - 1):
                dist = self.distance(int(hashes[all_index_list[j]]), int(hashes[all_index_list[j + 1]]))
                if dist <= nearby_min_dist:
                    del_index.add(all_index_list[j])
            all_index -= del_index

        # 2nd filter based on classes
        del_index = []
        all_dist = []
        for i in unique_labels:
            label_index = np.where(labels == i)
            label_index = label_index[0]
            label_index = set(label_index) & all_index
            label_index = list(label_index)
            for j in range(len(label_index) - 1):
                for k in range(j + 1, len(label_index)):
                    dist = self.distance(int(hashes[label_index[j]]), int(hashes[label_index[k]]))
                    if dist < class_min_dist:
                        class_min_dist = dist
                    if class_min_dist == 0:
                        break
                all_dist.append(class_min_dist)
                if class_min_dist == 0:
                    # to avoid over-filter, drop data by prob
                    if random.random() < over_filter_threshold:
                        del_index.append(label_index[j])

        # save filtered index
        del_index = set(del_index)
        left_index = all_index - del_index

        return list(left_index)

    def build_by_features(self, features):
        """build_by_features"""
        # basic vars
        sums, batch = [], []
        count = 0
        w = 1

        if isinstance(features, dict):
            features = features.items()

        for fea in features:
            skip_batch = False
            if not isinstance(fea, str):
                fea, w = fea
                skip_batch = w > self.large_weight_cutoff or not isinstance(w, int)

            count += w
            if isinstance(hashlib.md5(b"test").digest(), numbers.Integral):
                h = (hashlib.md5(fea.encode('utf-8')).digest() & self.truncate_mask).to_bytes(self.f_bytes, 'big')
            else:
                h = hashlib.md5(fea.encode('utf-8')).digest()[-self.f_bytes:]

            if skip_batch:
                sums.append(np.unpackbits(np.frombuffer(h, dtype='>B')) * w)
            else:
                batch.append(h * w)
                if len(batch) >= self.batch_size:
                    sums.append(sum_hashes(batch, self.f))
                    batch = []

            if len(sums) >= self.batch_size:
                sums = [np.sum(sums, 0)]

        if batch:
            sums.append(sum_hashes(batch, self.f))

        combined_sums = np.sum(sums, 0)
        value = int.from_bytes(np.packbits(combined_sums > count / 2).tobytes(), 'big')

        return value

    def distance(self, value1, value2):
        """Hamming distance"""
        x = (value1 ^ value2) & ((1 << self.f) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans


if __name__ == '__main__':
    # app_config_path = "./dataset/Hard_Pkt-8/Dataset/appConfig.csv"
    # train_path = "./dataset/Hard_Pkt-8/Dataset/Training_set"
    # device = "cpu"
    # config = read_config("./model_config.json")

    # simhash SA_ALL_filtered_normal_without_top
    train_data = pd.read_csv(
        "./dataset/Hard_Pkt-8-Saudi-0522/Training_set/APP_0/SA_ALL_filtered_normal_without_top-20240618172500-test-0.5part1.csv.bak")
    train_data["app_id"] = 0
    # train, test = train_test_split(train_data, test_size=0.5)
    # train.to_csv("./dataset/Hard_Pkt-8-Saudi-0522/Training_set/APP_0/SA_ALL_filtered_normal_without_top-20240618172500-test-0.5part1.csv.bak", index=False)
    # test.to_csv("./dataset/Hard_Pkt-8-Saudi-0522/Testing_set/APP_0/SA_ALL_filtered_normal_without_top-20240618172500-test-0.5part2.csv.bak", index=False)
    # id_to_appName = {0: "others"}
    # saudi_dataset = Saudi(train_data, device, config, id_to_appName)
    # simhash_indexes = simhash_process(saudi_dataset, sorted(id_to_appName.keys()))
    # simhash_train_data = train_data.iloc[simhash_indexes]
    # simhash_train_data.to_csv(
    #     "./dataset/Hard_Pkt-8-Saudi-0522/Testing_set/APP_0/SA_ALL_filtered_normal_without_top-20240618172500-test-simhashed.csv",
    #     index=False)
    train_data.sample(n=100000, replace=False).to_csv(
        "./dataset/Hard_Pkt-8-Saudi-0522/Training_set/APP_0/SA_ALL_filtered_normal_without_top-20240618172500-test-0.5part1-10w.csv.bak",
        index=False)

    filter = DataFilter()
    filter.run("./dataset/Hard_Pkt-8/Dataset/Testing_set", "./dataset/Hard_Pkt-8/Dataset/simhash_test")
