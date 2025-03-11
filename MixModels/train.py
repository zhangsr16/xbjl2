import multiprocessing
import os
import time
import sys

sys.path.append("./torch")
import numpy as np
import pandas as pd
import json

import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from pympler import asizeof

from data import get_data, read_data, read_env, Saudi
from utils import read_config, get_app_config_info, get_logger, log_confusion_matrix
from model import Model


class Trainer:
    """
    Trainer for the model
    """

    def __init__(self, app_config, config, input_dim, appName_to_id, id_to_appName, base_model_path, model_save_path,
                 logger=None,
                 model=None, params=None):
        self.app_config = app_config
        self.appName_to_id, self.id_to_appName = appName_to_id, id_to_appName
        self.base_model_path = base_model_path
        self.model_save_path = model_save_path
        self.input_dim = input_dim

        self.logger = get_logger() if logger is None else logger
        self.config = config
        self.device = torch.device(self.config["device"])

        if self.config["device"] == "cpu":
            torch.set_num_threads(min(multiprocessing.cpu_count(), self.config["trainer_cpu_cores"]))
            self.logger.info(f"Training cpu cores: {torch.get_num_threads()}")

        if model is None:
            self.model = Model(self.input_dim, len(self.appName_to_id), self.config).to(self.device)
        else:
            self.model = model.to(self.device)
        self.logger.info("=========================================================")
        self.logger.info("Model initialized successfully")
        self.logger.info("model config: %s", json.dumps(self.config, indent=4))
        self.logger.info("app_config: \n%s", app_config.to_string(index=False))
        if params is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["trainer_base_lr"],
                                               weight_decay=self.config["trainer_weight_decay"])
        else:
            self.optimizer = torch.optim.AdamW(params, lr=self.config["trainer_base_lr"],
                                               weight_decay=self.config["trainer_weight_decay"])
        self.criterion = torch.nn.NLLLoss()

    def inference(self, data_loader, others_threshold=None):
        """
        inference function.
        :param others_threshold: other apps' score threshold.
        :param data_loader:
        :return:
        """
        self.model.eval()
        predicts, labels, appendix_infos, pred_scores = [], [], [], []
        start = time.time()
        for inputs, label in data_loader:
            if not torch.is_tensor(label):
                label, appendix_info = label
                appendix_infos.append(appendix_info)

            out = self.model(inputs)
            pred_scores.append(out.detach().cpu().numpy())
            pred = out.argmax(1)
            predicts.append(pred)
            labels.append(label)

        predicts = torch.concat(predicts).detach().cpu().numpy()
        pred_scores = np.concatenate(pred_scores)
        if time.time() - start:
            self.logger.info("TEST SPEED: {:.2f} items/s".format(len(pred_scores) / (time.time() - start)))
        """
        When a threshold is specified, if the predicted score of the normal app is greater than this threshold,
        it will all be classified as the normal app. 
        """
        if others_threshold:
            predicts[pred_scores[:, self.appName_to_id["Others"]] >=
                     others_threshold] = self.appName_to_id["Others"]
        # if classifier is cosine.
        if hasattr(self.model, "tau"):
            pred_scores = pred_scores / self.model.tau.item()

        return predicts, torch.concat(labels).detach().cpu().numpy(), pred_scores, appendix_infos

    def finetune_initialize(self, new_class=0):
        """load the trained model params and reset the optimizer."""
        # 1.load model.
        model_loading_path = os.path.join(self.base_model_path, "model.pth")
        state_dict = torch.load(model_loading_path, map_location=self.device)
        if new_class > 0:
            original_output_dim = state_dict['output.weight'].size(0)
            assert original_output_dim + new_class == len(self.appName_to_id), \
                f"Loaded model class is {original_output_dim}, The appConfig has {len(self.appName_to_id)} class, " \
                f"new class is {new_class}."
            self.logger.info(f"Original model's output dim is {original_output_dim}")
            new_weight = torch.normal(torch.mean(state_dict['output.weight'], 0, keepdim=True).expand(new_class, -1),
                                      torch.std(state_dict['output.weight'], 0, keepdim=True).expand(new_class, -1))
            # new_weight = torch.randn(new_class, self.config["arch_hidden_dim"]).to(state_dict["output.weight"].device)
            # torch.nn.init.xavier_normal_(new_weight)
            state_dict["output.weight"] = torch.concat([state_dict["output.weight"], new_weight], dim=0)
            self.logger.info(f"Add {new_class} new classes, new output.weight is {state_dict['output.weight'].size()}")

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict)
        self.logger.info(f"Reloading the trained model from {model_loading_path} successfully. "
                         f"missing_keys:{missing_keys}, unexpected_keys:{unexpected_keys}")

        # 2.create new optimizer.
        cnn_param_names = ['payloads_to_hidden', 'ip_direction_pkt_len_to_hidden', 'mixed_to_hidden']
        cnn_params = list(self.model.payloads_to_hidden.parameters()) + \
                     list(self.model.ip_direction_pkt_len_to_hidden.parameters()) + \
                     list(self.model.mixed_to_hidden.parameters())
        other_params = [param for name, param in self.model.named_parameters() if
                        all(i not in name for i in cnn_param_names)]
        params = [{'params': cnn_params, 'lr': self.config["trainer_extractor_tuning_lr"]},
                  {'params': other_params, 'lr': self.config["trainer_classifier_tuning_lr"]}]
        self.optimizer = torch.optim.AdamW(params, lr=self.config["trainer_base_lr"],
                                           weight_decay=self.config["trainer_weight_decay"])

    def run(self, train_data, test_data, env_data, train_tensor, test_tensor, train_type="full", target_dict=None):
        """
        train function.
        :param train_data: train DataFrame
        :param test_data: test DataFrame
        :param train_type: train mode
            cold_start: train a new model based on the given data.
            full: Periodic retraining differs from a cold start in periodic retraining is based on an existing model
                and uses all types of data to retrain the model, effectively resetting the model to a certain extent.
            modification: There are two types. According to the instructions of target_dict, specific modification
                logic is executed for each type of data.
                a) increment: When a new class is introduced, mix the new class data with sampled data from other
                    classes and then perform fine-tuning.
                b) shift: When data drift occurs, mix the latest data of the drifted categories' data with historical
                    sampled data of all categories and then perform fine-tuning.
        :param target_dict: target_dict{AppId: task_type}
        :return: train loss, accuracy
        """
        self.logger.info(f"Training type: {train_type}")
        if train_type == "full":
            loading_path = os.path.join(self.base_model_path, "model.pth")
            self.model.load_state_dict(torch.load(loading_path, map_location=self.device))
            self.logger.info(f"Reloading the trained model from {loading_path} successfully")

        elif train_type == "modification":
            increment_app_ids, shift_app_ids, deleted_app_ids = set(), set(), set()
            for app_id, modification_type in target_dict.items():
                app_name = self.id_to_appName[app_id] if modification_type != "delete" else f"deletion app_id:{app_id}"
                if modification_type == "increment":
                    increment_app_ids.add(app_id)
                elif modification_type == "shift":
                    shift_app_ids.add(app_id)
                elif modification_type == "delete":
                    deleted_app_ids.add(app_id)
                else:
                    raise ValueError("Invalid modification type:{} for {}".format(modification_type, app_name))
            # deletion procedure.
            if deleted_app_ids:
                state_dict = torch.load(os.path.join(self.base_model_path, "model.pth"), map_location=self.device)
                ori_dim = state_dict['output.weight'].size(0)
                keep_columns = [i for i in range(state_dict['output.weight'].shape[0]) if i not in deleted_app_ids]
                state_dict['output.weight'] = state_dict['output.weight'][keep_columns, :]
                self.logger.info(f"Original model's output dim is {ori_dim}, delete {len(deleted_app_ids)} classes, "
                                 f"new output dim is {state_dict['output.weight'].size()}")

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict)
                self.logger.info(f"missing_keys:{missing_keys}, unexpected_keys:{unexpected_keys}")

                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)
                torch.save(state_dict, os.path.join(self.model_save_path, 'model.pth'))
                testset = Saudi(test_data, self.device, self.config, self.id_to_appName)
                test_loader = DataLoader(testset, batch_size=self.config["trainer_batchsize"])
                return self.evaluation(test_loader, sorted(self.id_to_appName.keys()), self.appName_to_id["Others"])

            train_data = train_data.sort_values(by='timestamp', ascending=False)  # sorted by time.
            modification_mask = train_data["app_id"].isin(increment_app_ids | shift_app_ids)
            # new train data sampling for both increment and drifted categories.
            new_train_data = train_data[modification_mask].groupby("app_id").apply(
                lambda x: x.head(self.config["trainer_tuning_new_samplesize"]))
            # old train data can't have any chosen new class's data.
            old_train_data = train_data.drop([i[1] for i in new_train_data.index])
            # old train data can't have any increment classes' data.
            if increment_app_ids:
                old_train_data = old_train_data[~old_train_data["app_id"].isin(increment_app_ids)]

            self.logger.info(
                "old_version data sampling method is {}".format(self.config["trainer_old_data_sample_method"]))

            if self.config["trainer_old_data_sample_method"] == "uniform":
                train_data = pd.concat(
                    [new_train_data, old_train_data.sample(
                        n=min(self.config["trainer_tuning_old_uniform_samplesize"], len(old_train_data)),
                        random_state=42, replace=False)])
            else:
                def balance_sampling(df):
                    """balance sampling for old version data of all classes"""
                    cur_app_id = df["app_id"].iloc[0]
                    # Don't need sampling for increment apps
                    if cur_app_id in increment_app_ids:
                        return
                    if len(df) <= self.config["trainer_tuning_old_balance_samplesize"]:
                        return df
                    else:
                        return df.sample(n=self.config["trainer_tuning_old_balance_samplesize"], random_state=42,
                                         replace=False)

                train_data = pd.concat([new_train_data, old_train_data.groupby("app_id").apply(balance_sampling)])

            self.finetune_initialize(new_class=len(increment_app_ids))

        elif train_type == "cold_start":
            pass  # do nothing.
        else:
            raise ValueError(f"Invalid train_type {train_type}")

        results = self.train(train_data, test_data, env_data, train_tensor, test_tensor)
        if train_type == "modification":
            for app_id, modification_type in target_dict.items():
                app_name = self.id_to_appName[app_id]
                if app_id in results["recalls"]:
                    self.logger.info("shift/increment class {} recall: {:.4f}, precision: {:.4f}, f1: {:.4f}".format(
                        app_name, results["recalls"][app_id], results["precisions"][app_id], results["f1s"][app_id]))
        return results

    def train(self, train_data, test_data, env_data, train_tensor, test_tensor):
        """
        train function.
        :param train_data: train DataFrame
        :param test_data: test DataFrame
        :return: train loss, accuracy
        """
        # create data set.
        balanced_trainset, testset = get_data(train_data, test_data, train_tensor, test_tensor, env_data, self.device,
                                              self.config, self.id_to_appName)
        # generated_data = load_generated_data_for_train(
        #     "./dataset/Hard_Pkt-8-Saudi-0522/generated_others/generated_others_other_th.npy", self.device, 100000)
        # balanced_trainset = balanced_trainset + generated_data

        # Calculate the total memory size occupied by dataset and all its attributes.
        total_data_size = asizeof.asizeof(balanced_trainset) + asizeof.asizeof(testset)
        self.logger.info(f"Total dataset memory usage in training procedure: {total_data_size / 1024 ** 2} MB, "
                         f"total items: {len(balanced_trainset) + len(testset)}")

        train_loader = DataLoader(balanced_trainset, batch_size=self.config["trainer_batchsize"], shuffle=True)
        test_loader = DataLoader(testset, batch_size=self.config["trainer_batchsize"])

        # train and validate.
        best_metric, no_bonus_count = 0, 0
        best_results = None

        for epoch in range(self.config['trainer_epoch']):
            # one epoch train.
            start = time.time()
            self.model.train()
            total, correct, train_loss = 0, 0, 0
            inputs = None
            for i, (inputs, labels) in enumerate(train_loader):
                if not torch.is_tensor(labels):
                    labels = labels[0]

                self.optimizer.zero_grad()
                out = self.model(inputs)
                loss = self.criterion(torch.log(out), labels)
                print("loss:", loss)
                loss.backward()
                self.optimizer.step()
                pred = out.argmax(1)

                # memory usage
                if i == 0:
                    if self.device.type == "cuda":
                        allocated_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2
                        self.logger.info(f"Epoch [{epoch + 1}/{self.config['trainer_epoch']}], "
                                         f"GPU memory allocated: {allocated_memory:.2f} MB")
                    else:
                        allocated_memory = asizeof.asizeof(self.model)
                        self.logger.info(f"Epoch [{epoch + 1}/{self.config['trainer_epoch']}], "
                                         f"memory allocated for model: {allocated_memory / 1024 ** 2} MB")

                correct += (pred == labels).sum()
                total += labels.size(0)
                train_loss += loss.item()

            train_acc = correct / total
            train_loss /= len(train_loader)
            self.logger.info("TRAIN SPEED: {:.2f} items/s".format(total / (time.time() - start)))

            results = self.evaluation(test_loader, sorted(self.id_to_appName.keys()), self.appName_to_id["Others"])

            if results[self.config["trainer_early_stopping_metric"]] > best_metric:
                best_metric = results[self.config["trainer_early_stopping_metric"]]
                best_results = results
                self.logger.info(
                    "New best {} found: {:.3f}".format(self.config['trainer_early_stopping_metric'], best_metric))
                # Save immediately if there is a better model.
                if self.model_save_path:
                    if not os.path.exists(self.model_save_path):
                        self.logger.info("Model saving directory not exists, try to make the dirs")
                        os.makedirs(self.model_save_path)
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'model.pth'))
                no_bonus_count = 0
            else:
                no_bonus_count += 1

            self.logger.info("epoch: %d, train_loss: %.3f, train_acc: %.3f" % (epoch, train_loss, train_acc))
            self.logger.info("test_acc: %.3f, top3_acc: %.3f, blocks_acc: %.3f" % (
                results["accuracy"], results["top3_accuracy"], results["blocks_acc"]))
            if 'bytes_cm' in results:
                self.logger.info("precision(macro), flow: {:.4f}, bytes: {:.4f}".format(results["macro_precision"],
                                                                                        results[
                                                                                            'macro_bytes_precision']))
                self.logger.info("recall(macro), flow: {:.4f}, bytes: {:.4f}".format(results["macro_recall"],
                                                                                     results['macro_bytes_recall']))
                self.logger.info("recall for NormalApps, flow: {:.4f}, bytes: {:.4f}".format(results["others_recall"],
                                                                                             results[
                                                                                                 'others_bytes_recall']))
                self.logger.info("recall for BlockApps, flow: {:.4f}, bytes: {:.4f}".format(results["blocks_recall"],
                                                                                            results[
                                                                                                'blocks_bytes_recall']))
            else:
                self.logger.info("precision(macro): {:.4f}".format(results["macro_precision"]))
                self.logger.info("recall(macro): {:.4f}".format(results["macro_recall"]))
                self.logger.info("recall for Normal: {:.4f}".format(results["others_recall"]))
                self.logger.info("recall for Block: {:.4f}".format(results["blocks_recall"]))

            log_confusion_matrix(self.logger, results["cm"], self.app_config)

            if no_bonus_count >= self.config["trainer_early_stopping_epoch"]:
                break

        return best_results

    def evaluation(self, data_loader, unique_labels, others_label):
        """
        evaluation function.
        :param data_loader:
        :param unique_labels: using for index the confusion matrix.
        :param others_label: other apps label id.
        :return:
        """
        predicts, labels, scores, appendix_infos = self.inference(data_loader)
        return evaluation(predicts, labels, scores, unique_labels, others_label, appendix_infos, self.app_config)


def evaluation(predicts, labels, pred_scores, unique_labels, others_label, appendix_infos=(), app_config=None):
    """
    Based on the given reasoning results, provide detailed evaluation metrics
    :param appendix_infos: Additional information for the calculation of some indicators.
    :param predicts: predicted labels.
    :param labels:  ground truth.
    :param pred_scores: predicted scores.
    :param unique_labels: using for index the confusion matrix.
    :param others_label: other apps label id.
    :param app_config: app info config dataframe.
    :return:
    """
    results = {}
    uniques = set(np.unique(labels))
    samples_max_label, given_max_label = max(uniques), max(unique_labels)
    if samples_max_label > given_max_label:
        raise ValueError(f"Max label in given labels is {samples_max_label}, "
                         f"but the unique_labels(given by app_config) is {given_max_label}")

    if appendix_infos:
        flow_bytes = np.concatenate([i.flow_bytes.detach().cpu().numpy() for i in appendix_infos])
        flow_bytes_cm = [[0] * (given_max_label + 1) for _ in range(given_max_label + 1)]
        for l, p, f in zip(labels, predicts, flow_bytes):
            flow_bytes_cm[l][p] += f
        flow_bytes_cm = np.array(flow_bytes_cm)

        bytes_precisions = []
        bytes_recalls = []
        for i in unique_labels:
            tp = flow_bytes_cm[i][i]
            fp = sum(flow_bytes_cm[:, i]) - tp
            fn = sum(flow_bytes_cm[i, :]) - tp
            bytes_precisions.append(tp / (tp + fp) if tp + fp != 0 else 0)
            bytes_recalls.append(tp / (tp + fn) if tp + fn != 0 else 0)

        others_total_bytes = flow_bytes_cm[others_label, :].sum()
        others_bytes_recall = flow_bytes_cm[others_label, others_label] / others_total_bytes
        blocks_bytes_recall = 1 - (
                flow_bytes_cm[:, others_label].sum() - flow_bytes_cm[others_label, others_label]) / (
                                      flow_bytes_cm.sum() - others_total_bytes)
        block_other_f1_bytes = 2 / (1 / others_bytes_recall + 1 / blocks_bytes_recall)

        results = {
            "macro_bytes_precision": np.mean(bytes_precisions),
            "macro_bytes_recall": np.mean(bytes_recalls),
            "bytes_cm": flow_bytes_cm,
            "others_bytes_recall": others_bytes_recall,
            "blocks_bytes_recall": blocks_bytes_recall,
            "block_other_f1_bytes": block_other_f1_bytes
        }

    top3_predicts = np.argsort(pred_scores)[:, -3:]
    top3_accuracy = np.mean(np.any(top3_predicts == labels[:, None], axis=1)).item()
    accuracy = np.mean(predicts == labels).item()

    cm = confusion_matrix(labels, predicts, labels=unique_labels)
    precisions = {l: precision_score(labels == l, predicts == l) for l in unique_labels if l in uniques}
    recalls = {l: recall_score(labels == l, predicts == l) for l in unique_labels if l in uniques}
    f1s = {l: f1_score(labels == l, predicts == l) for l in unique_labels if l in uniques}

    others_recall = cm[others_label, others_label] / cm[others_label, :].sum() if cm[others_label, :].sum() else 0
    if app_config is None:
        blocks_recall = 1 - (cm[:, others_label].sum() - cm[others_label, others_label]) / (
                cm.sum() - cm[others_label, :].sum())
        blocks_pred_count = cm.sum() - cm[:, others_label].sum()
        blocks_acc = (np.diag(cm).sum() - cm[
            others_label, others_label]) / blocks_pred_count if blocks_pred_count else 0
    else:
        block_labels = set(app_config["AppId"][app_config["IsBlocked"] == 1].values)
        total_block_count = sum(cm[bl, :].sum() for bl in block_labels)
        blocks_pred_count = sum(cm[:, bl].sum() for bl in block_labels)
        block_TP_count = sum(sum(c for pbl, c in enumerate(cm[bl, :]) if pbl in block_labels) for bl in block_labels)

        blocks_recall = block_TP_count / total_block_count if total_block_count else 0
        blocks_acc = sum(cm[bl, bl] for bl in block_labels) / blocks_pred_count if blocks_pred_count else 0

    block_other_f1 = 2 / (1 / others_recall + 1 / blocks_recall) if others_recall and blocks_recall else 0

    results["accuracy"] = accuracy
    results["top3_accuracy"] = top3_accuracy
    results["precisions"] = precisions
    results["recalls"] = recalls
    results["f1s"] = f1s
    results["macro_precision"] = sum(precisions.values()) / len(precisions)
    results["macro_recall"] = sum(recalls.values()) / len(recalls)
    results["macro_f1"] = sum(f1s.values()) / len(f1s)
    results["cm"] = cm
    results["others_recall"] = others_recall
    results["blocks_recall"] = blocks_recall
    results["blocks_acc"] = blocks_acc
    results["block_other_f1"] = block_other_f1

    return results


def main(train_path, test_path, env_path, app_config, train_type, base_model_path, model_save_path, logger=None,
         target_dict=None,
         folder_to_label=None):
    """train main"""
    if logger is None:
        logger = get_logger()
    logger.info("train_path: %s", train_path)
    logger.info("test_path: %s", test_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = read_config(os.path.join(script_dir, 'model_config.json'))
    appName_to_id, id_to_appName = get_app_config_info(app_config)

    # load the dataset
    train_data, train_tensor = read_data(config, train_path, id_to_appName, folder_to_label)
    test_data, test_tensor = read_data(config, test_path, id_to_appName, folder_to_label)
    env_data = read_env(config, env_path)
    hidden_dim = train_tensor.shape[1]
    for option in ['Area', 'Field']:
        cols_len = env_data[option].shape[1]
        kcols_len = len(config[option + 'Cluster'])
        cyc_len = env_data[option].shape[-1]
        hidden_dim += (cols_len + kcols_len) * cyc_len
    input_dim = {}
    input_dim['hidden_dim'] = hidden_dim
    input_dim['feature_dim'] = train_tensor.shape[-1] - 1
    trainer = Trainer(app_config, config, input_dim, appName_to_id, id_to_appName, base_model_path, model_save_path,
                      logger)
    results = trainer.run(train_data, test_data, env_data, train_tensor, test_tensor, train_type,
                          target_dict=target_dict)
    logger.info("Best {}: {:.4f}".format(trainer.config["trainer_early_stopping_metric"],
                                         results[trainer.config["trainer_early_stopping_metric"]]))
    logger.info("Best test_acc: %.3f, top3_acc: %.3f, blocks_acc: %.3f" % (
        results["accuracy"], results["top3_accuracy"], results["blocks_acc"]))
    logger.info("Best precision(macro), flow: {:.4f}".format(results["macro_precision"]))
    logger.info("Best recall(macro), flow: {:.4f}".format(results["macro_recall"]))
    logger.info("Best recall for Normal, flow: {:.4f}".format(results["others_recall"]))
    logger.info("Best recall for Block, flow: {:.4f}".format(results["blocks_recall"]))
    # logger.info("Best confusion matrix:\n%s", repr(results["cm"]))
    log_confusion_matrix(logger, results["cm"], app_config)


#### start of arch_hidden_range
# Area = 2
# "序号", "地区", "总交易额", "占市场", "股票交易额", "基金交易额", "债券交易额"
# Stock = 2
# "日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"
# Field = 2
# "项目名称", "项目名称-英文", "交易天数", "成交金额-人民币元", "成交金额-占总计", "成交股数-股数", "成交股数-占总计", "成交笔数-笔", "成交笔数-占总计"
# Summary = 1
# "单日情况", "股票", "主板A", "主板B", "科创板", "股票回购"
# Type = 1
# "证券类别", "数量", "成交金额", "总市值", "流通市值"
# THS = 4

if __name__ == '__main__':
    train_path = "./Training_set"
    test_path = "./Testing_set"
    env_path = "./Env_set"
    app_config_path = "./appConfig.csv"
    model_save_path = "./MODEL"
    base_model_path = ""
    train_type = "cold_start"  # "modification"
    target_dict = None  # {8: "increment", 9: "increment"}

    app_config_df = pd.read_csv(app_config_path).dropna(subset=['AppName'])  # 去除AppName==NA
    print(app_config_df)

    main(train_path, test_path, env_path, app_config_df, train_type, base_model_path, model_save_path,
         logger=get_logger(),
         target_dict=target_dict)
