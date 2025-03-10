import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Saudi, read_data
from train import Trainer, evaluation
from utils import read_config, get_app_config_info, get_logger, log_confusion_matrix
from model import Model


class Evaluator:
    """evaluation class"""

    def __init__(self, app_config, model_load_path, logger=None):
        self.app_config = app_config
        self.appName_to_id, self.id_to_appName = get_app_config_info(app_config)
        self.model_load_path = model_load_path
        self.logger = get_logger() if logger is None else logger
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = read_config(os.path.join(script_dir, 'model_config.json'))
        self.device = torch.device(self.config["device"])

        self.model = Model(len(self.appName_to_id), self.config).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.model_load_path, "model.pth"), map_location=self.device))
        self.model.to(self.device)

    def inference(self, data_set, threshold=None, inference_saving_path=None):
        """
        Inference the data_set.
        :param threshold: specified others recall's corresponding threshold.
        :param data_set:
        :param inference_saving_path: The path for saving the inference results as csv.
        :return:
        """
        # get data.
        data_saudi = Saudi(data_set, self.device, self.config, self.id_to_appName)
        data_loader = DataLoader(data_saudi, batch_size=self.config["trainer_batchsize"], shuffle=False)
        self.logger.info(f"Original dataset contained {len(data_set)} entries, and after processing,"
                         f" {len(data_saudi.data)} entries remained")
        # inference
        predicts, labels, pred_scores, appendix_infos = self.inference_dataloader(data_loader, threshold)
        flow_5tuples = np.array([j for i in appendix_infos for j in i.flow_5tuple])
        data_saudi.data["flow_5tuples"] = flow_5tuples
        data_saudi.data["predicts"] = [self.id_to_appName[i] for i in predicts]

        if inference_saving_path:
            if not os.path.exists(inference_saving_path):
                os.makedirs(inference_saving_path)
            data_saudi.data.to_csv(os.path.join(inference_saving_path, "results.csv"), index=False)
            self.logger.info(f"Result file successfully saved to the {inference_saving_path}")

        return predicts, labels, pred_scores, appendix_infos, data_saudi

    def inference_dataloader(self, data_loader, threshold=None):
        """
        Inference the data_loader.
        :param threshold: specified others recall's corresponding threshold.
        :param data_loader:
        :return:
        """
        trainer = Trainer(self.app_config, self.model_load_path, None, self.logger, self.model)
        return trainer.inference(data_loader, others_threshold=threshold)

    def inference_from_dir(self, data_dir, threshold=None, return_dataset=False):
        """
        Inference on the test set data in the dir.
        :param threshold: specified others recall's corresponding threshold.
        :param data_dir:
        :param return_dataset: return the saudi dataset or not.
        :return:
        """
        # get test data and trainer initialize.
        trainer = Trainer(self.app_config, self.model_load_path, None, self.logger, self.model)
        test_data = read_data(data_dir, trainer.id_to_appName)
        predicts, labels, pred_scores, appendix_infos, features, data_saudi = self.inference(test_data, threshold)
        if return_dataset:
            return predicts, labels, pred_scores, appendix_infos, features, data_saudi
        else:
            return predicts, labels, pred_scores, appendix_infos, features

    def run(self, data_set, train_type="full", target_dict=None):
        """
        evaluation function
        Infer on the test set data in the config.
        :param data_set: data csv file path.
        :param train_type: train mode
        :param target_dict: target_dict{AppId1..N: AppName1...N}
        :return:
        """
        self.logger.info(f"train type:{train_type}")
        predicts, labels, pred_scores, appendix_infos, _, _ = self.inference(data_set)
        results = evaluation(predicts, labels, pred_scores, sorted(self.id_to_appName.keys()),
                             self.appName_to_id["Others"], appendix_infos=appendix_infos)

        self.logger.info("test_acc: %.3f, top3_acc: %.3f, blocks_acc: %.3f" % (
            results["accuracy"], results["top3_accuracy"], results["blocks_acc"]))
        self.logger.info("precision(macro), flow: {:.4f}, bytes: {:.4f}".format(results["macro_precision"],
                                                                                results['macro_bytes_precision']))
        self.logger.info("recall(macro), flow: {:.4f}, bytes: {:.4f}".format(results["macro_recall"],
                                                                             results['macro_bytes_recall']))
        self.logger.info("recall for NormalApps, flow: {:.4f}, bytes: {:.4f}".format(results["others_recall"],
                                                                                     results['others_bytes_recall']))
        self.logger.info("recall for BlockApps, flow: {:.4f}, bytes: {:.4f}".format(results["blocks_recall"],
                                                                                    results['blocks_bytes_recall']))
        if target_dict:
            for app_id, modification_type in target_dict.items():
                if app_id in results["precisions"]:
                    app_name = self.id_to_appName[app_id]
                    results[f"{app_name}_flow_precision"] = results["precisions"][app_id]
                    results[f"{app_name}_flow_recall"] = results["recalls"][app_id]
                    results[f"{app_name}_bytes_precision"] = results["bytes_cm"][app_id, app_id] / results["bytes_cm"][:, app_id].sum()
                    results[f"{app_name}_bytes_recall"] = results["bytes_cm"][app_id, app_id] / results["bytes_cm"][app_id, :].sum()
                    self.logger.info(
                        "{} class {} flow recall: {:.4f}, flow precision: {:.4f}, flow f1: {:.4f}, bytes recall: {:.4f}, "
                        "bytes precision: {:.4f}".format(
                            modification_type, app_name, results["recalls"][app_id], results["precisions"][app_id],
                            results["f1s"][app_id], results[f"{app_name}_bytes_recall"],
                            results[f"{app_name}_bytes_precision"]))
                else:
                    self.logger.info(f"No data for app: {app_id}")

        # cm logger format
        log_confusion_matrix(self.logger, results['cm'], self.app_config)

        del results['bytes_cm'], results['cm'], results['precisions'], results['recalls'], results['f1s']

        return results


if __name__ == '__main__':
    app_config_path = "./appConfig.csv"
    model_load_path = './MODEL'

    # app_config
    app_config_df = pd.read_csv(app_config_path).dropna(subset=['AppName'])
    evaluator = Evaluator(app_config_df, model_load_path)
    # 1.inference from dir demo
    # predicts, labels, pred_scores, appendix_infos, features, ori = evaluator.inference_from_dir("C:/Users/z30060762/Desktop/AI协议识别/样本/GRE_Cold_Start_Base_0823/Testing_set", return_dataset=True)
    # 2.inference from csv
    data_df = pd.read_csv("./evaluation2.csv")
    # data_df = pd.read_csv("./dataset/Hard_Pkt-8-Saudi-0522+Hard_Pkt-8/Testing_set/APP_0/SA_ALL_filtered_normal_without_top-20240618172500-test-0.5part2.csv.bak")
    predicts, labels, pred_scores, appendix_infos, features = evaluator.inference(data_df)

    # 2.inference from dataloader
    # sa_all_simhash_others = np.load("./dataset/SA_ALL_filtered_normal_without_top-simhash.pkl")
    # sa_all_simhash_others = data_redirect(sa_all_simhash_others, "cuda")
    # predicts, labels, pred_scores, appendix_infos, features = evaluator.inference_dataloader(
    #     DataLoader(sa_all_simhash_others, batch_size=128))

    # redirect the normal predictions.
    # predicts[np.max(pred_scores, 1) <= 0.6] = 0trainer.inference(test_loader, others_threshold=threshold)
    # flow_5tuples = np.array([j for i in appendix_infos for j in i.flow_5tuple])
    # ori_data_features = [[j.detach().cpu().numpy().squeeze(0) for j in i[0]] for i in ori]
    # np.savez("./dataset/Hard_Pkt-8-Saudi-0522_for_exp117.npz", labels=labels, predict_labels=predicts,
    #          predict_scores=pred_scores, features=features, ori_features=ori_data_features)
    # print("test acc: {:.3f}".format(np.mean(predicts == labels).item()))

    # 2.metrics
    results = evaluation(predicts, labels, pred_scores, sorted(evaluator.id_to_appName.keys()),
                         evaluator.appName_to_id["Others"], appendix_infos=appendix_infos)

    print("test_acc: %.3f, top3_acc: %.3f, blocks_acc: %.3f" % (
        results["accuracy"], results["top3_accuracy"], results["blocks_acc"]))
    print("precision(macro), flow: {:.4f}, bytes: {:.4f}".format(results["macro_precision"],
                                                                 results['macro_bytes_precision']))
    print("recall(macro), flow: {:.4f}, bytes: {:.4f}".format(results["macro_recall"], results['macro_bytes_recall']))
    print("recall for NormalApps, flow: {:.4f}, bytes: {:.4f}".format(results["others_recall"],
                                                                      results['others_bytes_recall']))
    print("recall for BlockApps, flow: {:.4f}, bytes: {:.4f}".format(results["blocks_recall"],
                                                                     results['blocks_bytes_recall']))
    log_confusion_matrix(evaluator.logger, results["cm"], evaluator.app_config)
