from sklearn.metrics import roc_auc_score

config = {
    'input_dim': 200,
    'output_dim': 1,
    'translate_label_to_one_hot': False,
    'csv': 'PATH_TO_DATA/data/Santander_transaction/train.csv',

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
    df = df.drop('ID_code', axis=1)
    col = df.columns.tolist()
    col = col[1:] + [col[0]]
    df = df[col]
    return df