from Utils.metrics import multiclass_log_loss

config = {
    'input_dim': 129,
    'output_dim': 6,
    'translate_label_to_one_hot': True,
    'csv': 'PATH_TO_DATA/data/OpenML/Gas/gas.csv',

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
    map_labels = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5
    }
    df['Class'] = df['Class'].map(map_labels)
    return df