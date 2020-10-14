import os
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier


class XGBModelHandler:
    @staticmethod
    def train_and_test(config, data, score_config):
        print('train XGBoost')
        os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU']
        score_metric = score_config['score_metric']
        eval_metric = score_config['XGB_eval_metric']
        np.random.seed(seed=config['random_seed'])

        model = XGBClassifier(learning_rate=config['XGB_learning_rate'],
                              n_estimators=config['XGB_n_estimators'],
                              colsample_bytree=config['XGB_colsample_bytree'],
                              subsample=config['XGB_subsample'],
                              max_depth=config['XGB_max_depth'],
                              objective=config['XGB_objective'],
                              random_state=config['random_seed'],
                              gpu_id=0,
                              predictor='gpu_predictor',
                              tree_method='gpu_hist')

        eval_set = [(data['X_val'], data['Y_val'])]
        trained_model = model.fit(data['X_train'], data['Y_train'], eval_set=eval_set, eval_metric=eval_metric, verbose=False, early_stopping_rounds=50)

        if config['output_dim'] > 1:
            y_val = tf.keras.utils.to_categorical(data['Y_val'], num_classes=config['output_dim'])
            y_test = tf.keras.utils.to_categorical(data['Y_test'], num_classes=config['output_dim'])
        else:
            y_val = data['Y_val']
            y_test = data['Y_test']

        y_val_pred = trained_model.predict_proba(data['X_val'])
        if config['output_dim'] == 1:
            y_val_pred = y_val_pred[:, 1]
        validation_score = score_metric(y_val, y_val_pred)

        y_test_pred = trained_model.predict_proba(data['X_test'])
        if config['output_dim'] == 1:
            y_test_pred = y_test_pred[:, 1]
        test_score = score_metric(y_test, y_test_pred)

        n_epochs = trained_model.get_booster().best_iteration

        return {'test_score': test_score, 'validation_score': validation_score, 'n_epochs': n_epochs}
