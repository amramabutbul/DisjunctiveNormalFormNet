from Utils.metrics import multiclass_log_loss

config = {
    'input_dim': 32,
    'output_dim': 5,
    'translate_label_to_one_hot': True,
    'csv': 'PATH_TO_DATA/data/OpenML/gesture_phase/gesture_phase.csv',

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
        "'D'": 0,
        "'P'": 1,
        "'S'": 2,
        "'H'": 3,
        "'R'": 4,
    }
    df = df.replace({'Phase': map_labels})
    return df
