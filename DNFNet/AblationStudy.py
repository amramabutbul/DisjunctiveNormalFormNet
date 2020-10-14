import os
from DNFNet.CompetitionsHandler import CompetitionsHandler, shared_config
from DNFNet.ModelHandler import ModelHandler
from Utils.file_utils import create_dir
from Utils.grid_search_utils import cross_validation


# model2 - fully_trained_FCN
grid_params_exp_1 = {
    'n_formulas': [128, 512],   # 2048 is excluded due to out of memory
}

# model 3 - DNF_structure_only
grid_params_exp_2 = {
    'n_formulas': [128, 512, 2048],
}

# model 4 - DNF with feature selection
grid_params_exp_3 = {
    'n_formulas': [128, 512, 2048],
    'elastic_net_beta': [1.6, 1.3, 1., 0.7, 0.4, 0.1],
}

# model 1 - complete DNF-Net
grid_params_exp_4 = {
    'n_formulas': [128, 512, 2048],
    'elastic_net_beta': [1.6, 1.3, 1., 0.7, 0.4, 0.1],
}

# model 5 - DNF structure with localization (without feature selection)
grid_params_exp_5 = {
    'n_formulas': [128, 512, 2048],
}

# exp 6 - without localization - equal to exp 4

# model 6 - FCN with localization and feature selection
grid_params_exp_7 = {
    'n_formulas': [128, 512],  # 2048 is excluded due to out of memory
    'elastic_net_beta': [1.6, 1.3, 1., 0.7, 0.4, 0.1],
}


if __name__ == '__main__':
    base_dir = 'path/to/base/dir'
    model = CompetitionsHandler.DNFNET
    folds = [0, 1, 2, 3, 4]
    gpus_list = ['0']
    seeds_arr = [1, 2, 3]

    ablation_study_configs = [
                              {'exp_name': 'ablation_exp_1_fully_trained_FCN',
                               'model_number': 2,
                               'grid': grid_params_exp_1},

                              {'exp_name': 'ablation_exp_2_DNF_structure',
                               'model_number': 3,
                               'grid': grid_params_exp_2},

                              {'exp_name': 'ablation_exp_03_DNF_feature_selection',
                               'model_number': 4,
                               'grid': grid_params_exp_3},

                              {'exp_name': 'ablation_exp_04_DNF_structure_feature_selection_localization',
                               'model_number': 1,
                               'grid': grid_params_exp_4},

                              {'exp_name': 'ablation_exp_05_DNF-Net_without_feature_selection',
                               'model_number': 5,
                               'grid': grid_params_exp_5},

                              {'exp_name': 'ablation_exp_06_DNF-Net_without_DNF_structure',
                               'model_number': 6,
                               'grid': grid_params_exp_7}
                              ]

    for ablation_conf in ablation_study_configs:
        model_number = ablation_conf['model_number']
        grid_params = ablation_conf['grid']
        experiment_name = ablation_conf['exp_name']

        for competition_name in [CompetitionsHandler.GESTURE_PHASE, CompetitionsHandler.EYE_MOVEMENTS, CompetitionsHandler.GAS]:
            shared_config['experiments_dir'] = '{}/{}Competitions/{}/experiments/{}'.format(base_dir, model, competition_name, experiment_name)
            output_dir = os.path.join(shared_config['experiments_dir'], 'grid_search')
            create_dir(shared_config['experiments_dir'])

            shared_config['model_number'] = model_number
            shared_config['model_name'] = model
            shared_config['competition_name'] = competition_name
            config, score_config = CompetitionsHandler.get_configs(competition_name)
            config = CompetitionsHandler.merge_configs(shared_config, config)
            cross_validation(config, ModelHandler, score_config,
                             folds=folds,
                             gpus_list=gpus_list,
                             grid_params=grid_params,
                             output_dir=output_dir,
                             seeds_arr=seeds_arr)