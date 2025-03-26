import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Saudi, read_data, read_env
from train import Trainer, evaluation
from utils import read_config, get_app_config_info, get_logger, log_confusion_matrix
from model import Model


class Evaluator:
    """evaluation class"""

    def __init__(self, app_config, model_load_path, test_path, env_path, threshold=None, return_dataset=False,
                 logger=None):
        self.app_config = app_config
        self.appName_to_id, self.id_to_appName = get_app_config_info(app_config)
        self.model_load_path = model_load_path
        self.logger = get_logger() if logger is None else logger
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = read_config(os.path.join(script_dir, 'model_config.json'))
        self.device = torch.device(self.config["device"])

        self.test_data, self.test_tensor = read_data(self.config, test_path, self.id_to_appName)
        self.env_data = read_env(self.config, env_path)
        hidden_dim = self.test_data.shape[1]
        for option in ['Area', 'Field']:
            cols_len = self.env_data[option].shape[1]
            kcols_len = len(self.config[option + 'Cluster'])
            cyc_len = self.env_data[option].shape[-1]
            hidden_dim += (cols_len + kcols_len) * cyc_len
        self.input_dim = {'hidden_dim': hidden_dim, 'feature_dim': self.test_tensor.shape[-1] - 1}

        self.model = Model(self.input_dim, len(self.appName_to_id), self.config).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_load_path, "model.pth"), map_location=self.device))
        self.model.to(self.device)

    def inference(self, threshold=None, inference_saving_path=None):
        """
        Inference the data_set.
        :param threshold: specified others recall's corresponding threshold.
        :param data_set:
        :param inference_saving_path: The path for saving the inference results as csv.
        :return:
        """
        # get data.
        data_saudi = Saudi(self.test_data, self.test_tensor, self.env_data, self.device, self.config,
                           self.id_to_appName)
        data_loader = DataLoader(data_saudi, batch_size=self.config["trainer_batchsize"], shuffle=False)
        self.logger.info(f"Original dataset contained {len(self.test_data)} entries, and after processing,"
                         f" {len(data_saudi.data)} entries remained")
        # inference
        trainer = Trainer(self.app_config, self.config, self.input_dim, self.appName_to_id, self.id_to_appName,
                          self.model_load_path, self.model_load_path, self.logger, self.model)
        predicts, labels, pred_scores, appendix_infos = trainer.inference(data_loader, others_threshold=threshold)

        if inference_saving_path:
            if not os.path.exists(inference_saving_path):
                os.makedirs(inference_saving_path)
            data_saudi.data.to_csv(os.path.join(inference_saving_path, "results.csv"), index=False)
            self.logger.info(f"Result file successfully saved to the {inference_saving_path}")

        return predicts, labels, pred_scores, appendix_infos, data_saudi


if __name__ == '__main__':
    app_config_path = "./appConfig.csv"
    model_load_path = './MODEL'
    test_path = "./Predict_set"
    env_path = "./Env_set"
    # app_config
    app_config_df = pd.read_csv(app_config_path).dropna(subset=['AppName'])
    evaluator = Evaluator(app_config_df, model_load_path, test_path, env_path, )
    # inference from dir demo
    predicts, labels, pred_scores, appendix_infos, features = evaluator.inference()
    res_df = features.data[['DATE', '所属行业', '代码', '名称', 'app_name', 'app_id']]
    res_df['app_id'] = predicts
    res_df = res_df[res_df['app_id'] == 1]
    res_df = res_df.drop_duplicates(subset=['代码'])
    res_df = res_df.reset_index(drop=True)
    res_df = res_df.sort_values(by='代码')
    res_df.to_excel(res_df['DATE'][0] + '.xlsx', index=False)
    # 2.metrics
    results = evaluation(predicts, labels, pred_scores, sorted(evaluator.id_to_appName.keys()),
                         evaluator.appName_to_id["Others"], appendix_infos=appendix_infos)

    print("test_acc: %.3f, top3_acc: %.3f, blocks_acc: %.3f" % (
        results["accuracy"], results["top3_accuracy"], results["blocks_acc"]))
    print("precision(macro), flow: {:.4f}".format(results["macro_precision"]))
    print("recall(macro), flow: {:.4f}".format(results["macro_recall"], ))
    print("recall for NormalApps, flow: {:.4f}".format(results["others_recall"]))
    print("recall for BlockApps, flow: {:.4f}".format(results["blocks_recall"]))
    log_confusion_matrix(evaluator.logger, results["cm"], evaluator.app_config)
