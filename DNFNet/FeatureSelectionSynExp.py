import os
import numpy as np
from DNFNet.ModelHandler import ModelHandler
from Utils.file_utils import create_dir
from Utils.grid_search_utils import cross_validation
from data.data_utils import Ground_Truth_Mask_Generation
from sklearn.metrics import accuracy_score


def accuracy_wrapper(y_true, y_pred):
    y_pred_01 = np.zeros_like(y_true)
    y_pred_01[y_pred > 0.5] = 1
    return accuracy_score(y_true, y_pred_01)


config = {
    'model_number': 101,
    'model_name': 'FeatureSelectionSynExp',

    'output_dim': 1,
    'translate_label_to_one_hot': False,

    'initial_lr': 1e-3,
    'lr_decay_factor': 0.5,
    'lr_patience': 10,
    'min_lr': 1e-6,

    'early_stopping_patience': 30,
    'epochs': 1000,
    'batch_size': 256,

    'apply_standardization': True,

    'save_weights': True,
    'starting_epoch_to_save': 0,
    'models_module_name': 'DNFNet.DNFNetModels',
    'models_dir': './DNFNetModels',
}

score_config = {
    'score_metric': accuracy_wrapper,
    'score_increases': True,
}

batch_size = 256
FCN_grid = {'batch_size': [batch_size]}
FCN_with_oracle_mask_grid = {'batch_size': [batch_size]}
FCN_with_feature_selection_grid = {'batch_size': [batch_size], 'elastic_net_beta': [1.3, 1., 0.7, 0.4]}

grid_map = {'FCN': FCN_grid,
            'FCN_with_oracle_mask': FCN_with_oracle_mask_grid,
            'FCN_with_feature_selection': FCN_with_feature_selection_grid}

if __name__ == '__main__':
    base_dir = 'path/to/base/dir'
    model_type_arr = ['FCN', 'FCN_with_oracle_mask', 'FCN_with_feature_selection']
    syn_names = ['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']
    d_arr = [11, 50, 100, 150, 200, 250, 300]
    folds = [0, 1, 2, 3, 4]
    gpus_list = ['0']
    seeds_arr = [1, 2, 3]

    for syn in syn_names:
        for model_type in model_type_arr:
            for d in d_arr:
                experiment_name = 'exp_{}_model_{}_d_{}'.format(syn, model_type, d)
                config['experiments_dir'] = '{}/FeatureSelectionSynExp/experiments/{}'.format(base_dir, experiment_name)
                config['csv'] = 'data/FeatureSelectionSynExp/{}_{}/data.csv'.format(syn, d)
                config['input_dim'] = d
                config['mask'] = Ground_Truth_Mask_Generation(d, syn)
                config['model_type'] = model_type

                output_dir = os.path.join(config['experiments_dir'], 'grid_search')
                create_dir(config['experiments_dir'])

                cross_validation(config, ModelHandler, score_config,
                                 folds=folds,
                                 gpus_list=gpus_list,
                                 grid_params=grid_map[model_type],
                                 output_dir=output_dir,
                                 seeds_arr=seeds_arr)
