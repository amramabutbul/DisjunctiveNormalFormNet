from Utils.metrics import multiclass_log_loss

config = {
    'input_dim': 26,
    'output_dim': 3,
    'translate_label_to_one_hot': True,
    'csv': 'PATH_TO_DATA/data/OpenML/eye_movement/eye_movement.csv',

    'XGB_objective': 'multi:softmax',
}

score_config = {
    'score_metric': multiclass_log_loss,
    'score_increases': False,
    'XGB_eval_metric': 'mlogloss',
}


def get_configs():
    return config, score_config


def dataset_handler(df):
    df = df.drop('lineNo', axis=1)
    return df

