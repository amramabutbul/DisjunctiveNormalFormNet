import copy
import os
import time
from multiprocessing import Queue, Process, Manager

import numpy as np
import pandas as pd
from Utils.file_utils import delete_file, create_dir
from data.data_utils import read_train_val_test_folds
from scipy.stats import sem


def is_nan(x):
    return x is np.nan or x != x


def get_best_val_result(all_results, is_increasing):
    best_val_score = np.NINF if is_increasing else np.Inf
    best_val_experiment = None
    correspond_test_score = None
    for res in all_results:
        if res['validation_score'] is None:
            continue

        if is_increasing:
            if res['validation_score'] > best_val_score:
                best_val_score = res['validation_score']
                correspond_test_score = res['test_score']
                best_val_experiment = res['experiment_number']
        else:
            if res['validation_score'] < best_val_score:
                best_val_score = res['validation_score']
                correspond_test_score = res['test_score']
                best_val_experiment = res['experiment_number']
    return best_val_score, best_val_experiment, correspond_test_score


def training_worker(model_handler, config, train_args, return_dict):
    res = model_handler.train_and_test(config, **train_args)
    return_dict['res'] = res


def worker(model_handler, config, train_args, input_queue, output_queue, gpu_index):
    config['GPU'] = gpu_index

    while not input_queue.empty():
        try:
            c = input_queue.get_nowait()
        except Exception as e:
            if input_queue.empty():
                print('Done, no more configurations')
                exit(-1)
            else:
                continue

        for key, value in c.items():
            config[key] = value

        if config['model_name'] == 'DNFNet':
            if config['competition_name'] == 'Gas':
                if config['n_formulas'] == 3072:
                    config['batch_size'] = 1024
                else:
                    config['batch_size'] = 2048
            if config['competition_name'] == 'SantanderTransaction':
                if config['n_formulas'] == 3072 or config['n_formulas'] == 2048:
                    config['batch_size'] = 1024
                else:
                    config['batch_size'] = 2048

        manager = Manager()
        return_dict = manager.dict()
        p = Process(target=training_worker, args=(model_handler, config, train_args, return_dict))
        p.start()
        p.join()
        res = return_dict['res']
        output_queue.put({'res': res, 'config': c})

    print('Done, no more configurations')
    exit(-1)


def some_worker_is_alive(worker_list):
    for p in worker_list:
        if p.is_alive():
            return True
    return False


def create_all_permutations(grid_params, permutations):
    if len(grid_params) == 0:
        return permutations
    key = next(iter(grid_params))
    param_list = grid_params[key]
    del grid_params[key]
    new_permutations = []
    for p in permutations:
        for val in param_list:
            new_p = p.copy()
            new_p[key] = val
            new_permutations.append(new_p)
    return create_all_permutations(grid_params, new_permutations)


def validate_results(df):
    for _, res in df.iterrows():
        if is_nan(res['validation_score']) or is_nan(res['test_score']):
            return False
    return True


def distributed_grid_search(GPUs, model_handler, config, grid_parmas, name, train_args, output_dir):
    all_configurations = create_all_permutations(grid_parmas, [{}])
    input_queue = Queue(maxsize=len(all_configurations) + 1)
    output_queue = Queue(maxsize=len(all_configurations) + 1)
    process_list = []

    exp_i = config['experiment_number']
    for c in all_configurations:
        c['experiment_number'] = exp_i
        input_queue.put(c)
        exp_i += 1

    for i in range(len(GPUs)):
        p = Process(target=worker, args=(model_handler, config.copy(), train_args, input_queue, output_queue, GPUs[i]))
        time.sleep(1)  # for queue issues
        p.start()
        process_list.append(p)

    all_results = []
    csv_lines = []
    i = 1
    cols = list(all_configurations[0].keys())
    while not output_queue.empty() or (len(all_results) != len(all_configurations) and some_worker_is_alive(process_list)):
        res = output_queue.get()
        res['res']['experiment_number'] = res['config']['experiment_number']
        all_results.append(res['res'])
        res_i = dict()
        for key, value in res['res'].items():
            res_i[key] = value
        for key, value in res['config'].items():
            res_i[key] = value
        csv_lines.append(res_i)
        df = pd.DataFrame(csv_lines)
        df.to_csv('{}/{}_{}.csv'.format(output_dir, name, i), columns=cols + list(res['res'].keys()))
        delete_file('{}/{}_{}.csv'.format(output_dir, name, i - 1))
        i += 1

    for p in process_list:
        p.join()

    is_valid = validate_results(df)
    return all_results, is_valid


def model_handler_params(data, score_config):
    return {'data': data, 'score_config': score_config}


def write_cv_results(all_test_scores, all_val_scores, all_test_experiments, output_dir, name=''):
    print('all test scores:')
    test_res_csv_rows = []
    for test_score, val_score, exp_number in zip(all_test_scores, all_val_scores, all_test_experiments):
        print('experiment number: {}, test score: {}'.format(exp_number, test_score))
        test_res_csv_rows.append({
            'experiment': exp_number,
            'test_score': test_score,
            'val_score': val_score,
            'mean test': None,
            'sem test': None,
            'mean val': None,
            'sem val': None,
        })

    test_scores_mean = np.mean(all_test_scores)
    test_scores_sem = sem(all_test_scores)
    print('mean of test scores: {}'.format(test_scores_mean))
    print('sem: {}'.format(test_scores_sem))

    test_res_csv_rows.append({
        'experiment': None,
        'test_score': None,
        'val_score': None,
        'mean test': test_scores_mean,
        'sem test': test_scores_sem,
        'mean val': np.mean(all_val_scores),
        'sem val': sem(all_val_scores),
    })

    df = pd.DataFrame(test_res_csv_rows)
    df.to_csv(os.path.join(output_dir, 'test_scores{}.csv'.format(name)))


def cross_validation(config, model_handler, score_config, folds, gpus_list, grid_params, seeds_arr, output_dir='./grid_search'):
    create_dir(output_dir)

    if 'experiment_number' not in config:
        config['experiment_number'] = 0

    valid_list = []
    for seed in seeds_arr:
        all_test_scores = []
        all_val_scores = []
        all_test_experiments = []
        for k in folds:
            config['random_seed'] = seed
            data = read_train_val_test_folds(config['csv'], k, apply_standardization=config['apply_standardization'])

            all_results, is_valid = distributed_grid_search(gpus_list, model_handler, copy.deepcopy(config),
                                                            grid_parmas=copy.deepcopy(grid_params),
                                                            name='iter_{}_seed_{}_conf'.format(k, seed),
                                                            train_args=model_handler_params(data, score_config),
                                                            output_dir=output_dir)
            config['experiment_number'] += len(all_results)

            best_val_score, best_val_experiment, correspond_test_score = get_best_val_result(all_results, score_config['score_increases'])
            all_test_scores.append(correspond_test_score)
            all_val_scores.append(best_val_score)
            all_test_experiments.append(best_val_experiment)
            valid_list.append({'fold': k, 'seed': seed, 'is valid': is_valid})

        write_cv_results(all_test_scores, all_val_scores, all_test_experiments, output_dir, name='_{}'.format(str(seed)))

    for valid_desc in valid_list:
        print(valid_desc)
