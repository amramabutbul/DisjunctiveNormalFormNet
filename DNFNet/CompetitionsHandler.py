import copy
import os

from DNFNet.Competitions.EyeMovements.CompetitionConfig import get_configs as get_eye_movements_configs
from DNFNet.Competitions.Gas.CompetitionConfig import get_configs as get_gas_configs
from DNFNet.Competitions.GesturePhase.CompetitionConfig import get_configs as get_gesture_phase_configs
from DNFNet.Competitions.Otto.CompetitionConfig import get_configs as get_otto_configs
from DNFNet.Competitions.SantanderTransaction.CompetitionConfig import get_configs as get_santander_transaction_configs
from DNFNet.Competitions.House.CompetitionConfig import get_configs as get_house_configs
from DNFNet.Competitions.RobotNavigation.CompetitionConfig import get_configs as get_robot_navigation_configs


from DNFNet.ModelHandler import ModelHandler
from DNFNet.XGBModelHandler import XGBModelHandler

from Utils.experiment_utils import create_all_FCN_layers_grid
from Utils.file_utils import create_dir
from Utils.grid_search_utils import cross_validation


class CompetitionsHandler:
    # Competitions
    GAS = 'Gas'
    OTTO = 'Otto'
    EYE_MOVEMENTS = 'EyeMovements'
    GESTURE_PHASE = 'GesturePhase'
    SANTANDER_TRANSACTION = 'SantanderTransaction'
    HOUSE = 'House'
    ROBOT_NAVIGATION = 'RobotNavigation'

    # Models
    DNFNET = 'DNFNet'
    FCN = 'FCN'
    XGB = 'XGB'

    @staticmethod
    def get_configs(competition_name):
        if competition_name == CompetitionsHandler.EYE_MOVEMENTS:
            return get_eye_movements_configs()
        elif competition_name == CompetitionsHandler.GAS:
            return get_gas_configs()
        elif competition_name == CompetitionsHandler.GESTURE_PHASE:
            return get_gesture_phase_configs()
        elif competition_name == CompetitionsHandler.OTTO:
            return get_otto_configs()
        elif competition_name == CompetitionsHandler.SANTANDER_TRANSACTION:
            return get_santander_transaction_configs()
        elif competition_name == CompetitionsHandler.HOUSE:
            return get_house_configs()
        elif competition_name == CompetitionsHandler.ROBOT_NAVIGATION:
            return get_robot_navigation_configs()


    @staticmethod
    def merge_configs(config_dst, config_src):
        config_dst = copy.deepcopy(config_dst)
        config_src = copy.deepcopy(config_src)
        for key, value in config_src.items():
            config_dst[key] = value
        return config_dst


DNFNet_grid_params = {
    'n_formulas': [3072, 2048, 1024, 512, 256, 128, 64],
    'orthogonal_lambda': [0.],
    'elastic_net_beta': [1.6, 1.3, 1., 0.7, 0.4, 0.1],
}

FCN_grid_params = {
    'FCN_layers': create_all_FCN_layers_grid(depth_arr=[1, 2, 3, 4, 5, 6], width_arr=[128, 256, 512, 1024, 2048]),
    'FCN_l2_lambda': [1e-2, 1e-4, 1e-6, 1e-8, 0.],
    'dropout_rate': [0., 0.25, 0.5, 0.75],
    'initial_lr': [5e-2, 5e-3, 5e-4]
}

XGB_grid_params = {
    'XGB_n_estimators': [2500],
    'XGB_learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    'XGB_max_depth': [2, 3, 4, 5, 7, 9, 11, 13, 15],
    'XGB_colsample_bytree': [0.25, 0.5, 0.75, 1.],
    'XGB_subsample': [0.25, 0.5, 0.75, 1.]
}

shared_config = {
    'n_conjunctions_arr': [6, 9, 12, 15],
    'conjunctions_depth_arr': [2, 4, 6],
    'keep_feature_prob_arr': [0.1, 0.3, 0.5, 0.7, 0.9],

    'initial_lr': 5e-2,
    'lr_decay_factor': 0.5,
    'lr_patience': 10,
    'min_lr': 1e-6,

    'early_stopping_patience': 30,
    'epochs': 1000,
    'batch_size': 2048,

    'apply_standardization': True,

    'save_weights': True,
    'starting_epoch_to_save': 0,
    'models_module_name': 'DNFNet.DNFNetModels',
    'models_dir': './DNFNetModels',
}


if __name__ == '__main__':
    competition_name_list = [
                             CompetitionsHandler.SANTANDER_TRANSACTION,
                             CompetitionsHandler.OTTO,
                             CompetitionsHandler.GESTURE_PHASE,
                             CompetitionsHandler.EYE_MOVEMENTS,
                             CompetitionsHandler.GAS,
                             CompetitionsHandler.HOUSE,
                             CompetitionsHandler.ROBOT_NAVIGATION
                            ]

    base_dir = 'path/to/base/dir'
    model = CompetitionsHandler.FCN
    experiment_name = 'exp_comparative_evaluation'
    folds = [0, 1, 2, 3, 4]
    gpus_list = ['0']
    seeds_arr = [1, 2, 3]

    for competition_name in competition_name_list:
        shared_config['experiments_dir'] = '{}/{}Competitions/{}/experiments/{}'.format(base_dir, model, competition_name, experiment_name)
        output_dir = os.path.join(shared_config['experiments_dir'], 'grid_search')
        create_dir(shared_config['experiments_dir'])
        shared_config['competition_name'] = competition_name
        shared_config['model_name'] = model

        print(competition_name)
        if model == CompetitionsHandler.DNFNET:
            shared_config['model_number'] = 1
            config, score_config = CompetitionsHandler.get_configs(competition_name)
            config = CompetitionsHandler.merge_configs(shared_config, config)
            cross_validation(config, ModelHandler, score_config,
                             folds=folds,
                             gpus_list=gpus_list,
                             grid_params=DNFNet_grid_params,
                             output_dir=output_dir,
                             seeds_arr=seeds_arr)

        elif model == CompetitionsHandler.FCN:
            shared_config['model_number'] = 100
            config, score_config = CompetitionsHandler.get_configs(competition_name)
            config = CompetitionsHandler.merge_configs(shared_config, config)
            cross_validation(config, ModelHandler, score_config,
                             folds=folds,
                             gpus_list=gpus_list,
                             grid_params=FCN_grid_params,
                             output_dir=output_dir,
                             seeds_arr=seeds_arr)

        elif model == CompetitionsHandler.XGB:
            config, score_config = CompetitionsHandler.get_configs(competition_name)
            config = CompetitionsHandler.merge_configs(shared_config, config)
            cross_validation(config, XGBModelHandler, score_config,
                             folds=folds,
                             gpus_list=gpus_list,
                             grid_params=XGB_grid_params,
                             output_dir=output_dir,
                             seeds_arr=seeds_arr)
