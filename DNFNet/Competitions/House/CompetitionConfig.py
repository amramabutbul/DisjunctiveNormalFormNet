from sklearn.metrics import roc_auc_score

config = {
    'input_dim': 16,
    'output_dim': 1,
    'translate_label_to_one_hot': False,
    'csv': 'PATH_TO_DATA/data/OpenML/House/house.csv',

    'XGB_objective': "binary:logistic",
}

score_config = {
    'score_metric': roc_auc_score,
    'score_increases': True,
    'XGB_eval_metric': 'auc',
}


def get_configs():
    return config, score_config


def dataset_handler(df):
    map_labels = {
        "N": 0,
        "P": 1,
    }
    df = df.replace({'binaryClass': map_labels})
    return df
