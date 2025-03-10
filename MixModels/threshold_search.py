import os

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from evaluation import Evaluator
from utils import read_config, get_app_config_info
from data import read_data


def threshold_search(scores, labels, target_labels, target_recall, level="flow", flow_bytes=(), eps=1e-3):
    """
    threshold value searching algorithm.
    :param level: The basis of threshold search. options: 'flow' or 'bytes'.
    :param flow_bytes: Additional information for the calculation of some indicators.
    :param eps:
    :param labels:
    :param scores:
    :param target_labels:
    :param target_recall:
    """
    low, high = 0, 1
    best_threshold = None
    best_threshold_recall = None
    target_mask = np.in1d(labels, target_labels)
    target_label_total_bytes = np.sum(flow_bytes[target_mask]) if len(flow_bytes) else None

    while high - low > eps:
        mid = (low + high) / 2
        # If the predicted score of the normal app is less than the threshold,
        # the predicted category will be given according to argmax, and it is still possible to predict as a normal app.
        predictions = np.argmax(scores, axis=-1)
        for tl in target_labels:
            predictions[scores[:, tl] >= mid] = tl

        predictions = np.in1d(predictions, target_labels)

        if level == "flow":
            recall_val = recall_score(target_mask, predictions)
        else:
            target_total_bytes_TP = flow_bytes[predictions & target_mask].sum()
            recall_val = target_total_bytes_TP / target_label_total_bytes

        if recall_val >= target_recall:
            best_threshold = mid
            best_threshold_recall = recall_val
            # Try a larger threshold to meet the condition.
            low = mid
        else:
            # Need a smaller threshold to increase the recall.
            high = mid
    if best_threshold is None:
        raise ValueError("Warning: Unable to find a suitable threshold while meeting the target recall,"
                         " please try a smaller eps.")

    return best_threshold, best_threshold_recall


def get_threshold(app_config, model_path, data_set, logger=None):
    """
    Search for the corresponding threshold based on the specified recall
    :param app_config: app config info dataframe.
    :param model_path:
    :param data_set:
    :param logger:
    :return:
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configuration = read_config(os.path.join(script_dir, 'model_config.json'))

    predicts, ground_truths, pred_scores, appendix_infos, _ = Evaluator(app_config, model_path, logger).inference(data_set)
    flow_bytes = np.concatenate([i[1].detach().cpu().numpy() for i in appendix_infos])
    normal_ids = app_config["AppId"][app_config["IsBlocked"] == 0].values

    threshold, recall = threshold_search(pred_scores, ground_truths, normal_ids,
                                         configuration["eval_target_others_recall"],
                                         level=configuration["eval_target_others_recall_level"],
                                         flow_bytes=flow_bytes)
    return threshold


if __name__ == '__main__':
    # app_config
    app_config_df = pd.read_csv("./dataset/Hard_Pkt-8-Saudi-0522/appConfig.csv").dropna(subset=['AppName'])
    appName_to_id, id_to_appName = get_app_config_info(app_config_df)

    # load the dataset
    test_data = read_data("./dataset/Hard_Pkt-8-Saudi-0522/Testing_set", id_to_appName)
    threshold = get_threshold(app_config_df, "./model/Increment_test", test_data)
    print(threshold)
    # print(get_threshold_from_file(result_path="./config/results.json", target_recall=config['eval_target_others_recall']))
