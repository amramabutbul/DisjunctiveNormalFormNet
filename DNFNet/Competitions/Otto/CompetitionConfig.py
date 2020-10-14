from Utils.metrics import multiclass_log_loss

config = {
    'input_dim': 93,
    'output_dim': 9,
    'translate_label_to_one_hot': True,
    'csv': 'PATH_TO_DATA/data/Otto/train.csv',

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
        'Class_1': 0,
        'Class_2': 1,
        'Class_3': 2,
        'Class_4': 3,
        'Class_5': 4,
        'Class_6': 5,
        'Class_7': 6,
        'Class_8': 7,
        'Class_9': 8,
    }
    df = df.replace({'target': map_labels})
    df = df.drop('id', axis=1)
    return df