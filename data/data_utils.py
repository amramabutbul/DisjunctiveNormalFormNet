import os
import pandas as pd
import numpy as np
from DNFNet.Competitions.Otto.CompetitionConfig import dataset_handler as otto_dataset_handler
from DNFNet.Competitions.SantanderTransaction.CompetitionConfig import dataset_handler as santander_transaction_dataset_handler
from DNFNet.Competitions.EyeMovements.CompetitionConfig import dataset_handler as eye_movements_dataset_handler
from DNFNet.Competitions.GesturePhase.CompetitionConfig import dataset_handler as gesture_phase_dataset_handler
from DNFNet.Competitions.Gas.CompetitionConfig import dataset_handler as gas_dataset_handler
from DNFNet.Competitions.House.CompetitionConfig import dataset_handler as house_dataset_handler
from DNFNet.Competitions.RobotNavigation.CompetitionConfig import dataset_handler as robot_dataset_handler

from Utils.file_utils import create_dir
from sklearn.model_selection import StratifiedKFold, train_test_split


def read_train_val_test_folds(csv_path, fold_idx, apply_standardization=True):
    fold_dir = os.path.join(os.path.dirname(csv_path), 'cross_validation', 'fold_{}'.format(fold_idx))
    train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(fold_dir, 'test.csv'))

    X_train = train_df.iloc[:, :-1].values
    Y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values
    Y_val = val_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    Y_test = test_df.iloc[:, -1].values

    if apply_standardization:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

    data = dict()
    data['X_train'] = X_train.astype('float32')
    data['X_val'] = X_val.astype('float32')
    data['X_test'] = X_test.astype('float32')
    data['Y_train'] = Y_train.astype('float32')
    data['Y_val'] = Y_val.astype('float32')
    data['Y_test'] = Y_test.astype('float32')
    return data


class DatasetHandler:
    # Competitions
    GAS = 'Gas'
    OTTO = 'Otto'
    EYE_MOVEMENTS = 'EyeMovements'
    GESTURE_PHASE = 'GesturePhase'
    SANTANDER_TRANSACTION = 'SantanderTransaction'
    HOUSE = 'House'
    ROBOT_NAVIGATION = 'RobotNavigation'


    @staticmethod
    def get_dataset_handler(competition_name):
        if competition_name == DatasetHandler.OTTO:
            return otto_dataset_handler
        elif competition_name == DatasetHandler.SANTANDER_TRANSACTION:
            return santander_transaction_dataset_handler
        elif competition_name == DatasetHandler.EYE_MOVEMENTS:
            return eye_movements_dataset_handler
        elif competition_name == DatasetHandler.GESTURE_PHASE:
            return gesture_phase_dataset_handler
        elif competition_name == DatasetHandler.GAS:
            return gas_dataset_handler
        elif competition_name == DatasetHandler.HOUSE:
            return house_dataset_handler
        elif competition_name == DatasetHandler.ROBOT_NAVIGATION:
            return robot_dataset_handler


def create_dataset_partitions(csv_path, dataset_name=None, train_proportion=0.7, k_folds=5, seed=1):
    print(csv_path)
    np.random.seed(seed=seed)
    output_dir = os.path.dirname(csv_path) + '/cross_validation'
    create_dir(output_dir)
    create_dir(output_dir + '/StratifiedKFold')
    create_dir(output_dir + '/seed_{}'.format(seed))

    df = pd.read_csv(csv_path)
    if dataset_name is not None:
        dataset_handler = DatasetHandler.get_dataset_handler(dataset_name)
        df = dataset_handler(df)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    y = df.iloc[:, -1].values
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    val_proportion = 1 - (1. / k_folds) - train_proportion
    for i, (train_val_idx, test_idx) in enumerate(kf.split(df, y)):
        df_i = df.copy()
        test_df = df_i.iloc[test_idx]
        train_val_df = df_i.iloc[train_val_idx]
        y_train_val = train_val_df.iloc[:, -1].values
        train_df, val_df, _, _ = train_test_split(train_val_df, y_train_val, random_state=seed, stratify=y_train_val,
                                                  test_size=val_proportion / (val_proportion + train_proportion))

        fold_dir = os.path.join(output_dir, 'fold_{}'.format(i))
        create_dir(fold_dir)
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)


################################################################################################################################################################################################
#  Feature selection - syn data generation
################################################################################################################################################################################################
'''
Based on the work of 'Jinsung Yoon': https://github.com/jsyoon0823/INVASE/blob/master/data_generation.py


Written by Jinsung Yoon
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "IINVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu
---------------------------------------------------
Generating Synthetic Data for Synthetic Examples
There are 6 Synthetic Datasets
X ~ N(0,I) where d = 100
Y = 1/(1+logit)
- Syn1: logit = exp(X1 * X2)
- Syn2: logit = exp(X3^2 + X4^2 + X5^2 + X6^2 -4)
- Syn3: logit = -10 sin(2 * X7) + 2|X8| + X9 + exp(-X10) - 2.4
- Syn4: If X11 < 0, Syn1, X11 >= Syn2
- Syn5: If X11 < 0, Syn1, X11 >= Syn3
- Syn6: If X11 < 0, Syn2, X11 >= Syn3
'''


# %% Basic Label Generation (Syn1, Syn2, Syn3)
def Basic_Label_Generation(X, data_type):
    # number of samples
    n = len(X[:, 0])

    # Logit computation
    # 1. Syn1
    if data_type == 'Syn1':
        logit = np.exp(X[:, 0] * X[:, 1])

    # 2. Syn2
    elif data_type == 'Syn2':
        logit = np.exp(np.sum(X[:, 2:6] ** 2, axis=1) - 4.0)

    # 3. Syn3
    elif data_type == 'Syn3':
        logit = np.exp(-10 * np.sin(0.2 * X[:, 6]) + abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9]) - 2.4)

    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape((1 / (1 + logit)), [n, 1])
    prob_0 = np.reshape((logit / (1 + logit)), [n, 1])

    # Probability output
    prob_y = np.concatenate((prob_0, prob_1), axis=1)

    # Sampling from the probability
    y = np.zeros([n, 2])
    y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n, ])
    y[:, 1] = 1 - y[:, 0]

    return y[:, 1]


# %% Complex Label Generation (Syn4, Syn5, Syn6)
def Complex_Label_Generation(X, data_type):
    # number of samples
    n = len(X[:, 0])

    # Logit generation
    # 1. Syn4
    if data_type == 'Syn4':
        logit1 = np.exp(X[:, 0] * X[:, 1])
        logit2 = np.exp(np.sum(X[:, 2:6] ** 2, axis=1) - 4.0)

    # 2. Syn5
    elif data_type == 'Syn5':
        logit1 = np.exp(X[:, 0] * X[:, 1])
        logit2 = np.exp(-10 * np.sin(0.2 * X[:, 6]) + abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9]) - 2.4)

    # 3. Syn6
    elif data_type == 'Syn6':
        logit1 = np.exp(np.sum(X[:, 2:6] ** 2, axis=1) - 4.0)
        logit2 = np.exp(-10 * np.sin(0.2 * X[:, 6]) + abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9]) - 2.4)

    # Based on X[:,10], combine two logits
    idx1 = (X[:, 10] < 0) * 1
    idx2 = (X[:, 10] >= 0) * 1

    logit = logit1 * idx1 + logit2 * idx2

    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape((1 / (1 + logit)), [n, 1])
    prob_0 = np.reshape((logit / (1 + logit)), [n, 1])

    # Probability output
    prob_y = np.concatenate((prob_0, prob_1), axis=1)

    # Sampling from the probability
    y = np.zeros([n, 2])
    y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n, ])
    y[:, 1] = 1 - y[:, 0]

    return y[:, 1]


# %% Ground truth Variable Importance
def Ground_Truth_Mask_Generation(n_features, data_type):
    # mask initialization
    m = np.zeros(n_features)

    # For each data_type
    # Simple
    if data_type in ['Syn1', 'Syn2', 'Syn3']:
        if data_type == 'Syn1':
            m[:2] = 1
        elif data_type == 'Syn2':
            m[2:6] = 1
        elif data_type == 'Syn3':
            m[6:10] = 1

    # Complex
    if data_type in ['Syn4', 'Syn5', 'Syn6']:
        if data_type == 'Syn4':
            m[:2] = 1
            m[2:6] = 1
        elif data_type == 'Syn5':
            m[:2] = 1
            m[6:10] = 1
        elif data_type == 'Syn6':
            m[2:6] = 1
            m[6:10] = 1
        m[10] = 1
    return m


# %% Generate X and Y
def generate_data(n=10000, d=11, data_type='Syn1', seed=1, output_dir='.'):
    """
    :param n: Number of samples
    :param d: input dimension
    :param data_type: the name of the syn dataset
    :param seed: random seed for numpy
    """

    np.random.seed(seed)

    # X generation
    X = np.random.randn(n, d)

    # Y generation
    if data_type in ['Syn1', 'Syn2', 'Syn3']:
        Y = Basic_Label_Generation(X, data_type)

    elif data_type in ['Syn4', 'Syn5', 'Syn6']:
        Y = Complex_Label_Generation(X, data_type)

    data = np.concatenate([X, np.expand_dims(Y, axis=1)], axis=1)
    output_dir = os.path.join(output_dir, '{}_{}'.format(data_type, str(d)))
    create_dir(output_dir)
    output_path = os.path.join(output_dir, 'data.csv')
    pd.DataFrame(data=data).to_csv(output_path, index=False)
    create_dataset_partitions(output_path)


if __name__ == '__main__':
    create_dataset_partitions('PATH_TO_DATA/data/Otto/train.csv', dataset_name=DatasetHandler.OTTO)
    create_dataset_partitions('PATH_TO_DATA/data/SantanderTransaction/train.csv', dataset_name=DatasetHandler.SANTANDER_TRANSACTION)
    create_dataset_partitions('PATH_TO_DATA/data/OpenML/EyeMovements/EyeMovements.csv', dataset_name=DatasetHandler.EYE_MOVEMENTS)
    create_dataset_partitions('PATH_TO_DATA/data/OpenML/GesturePhase/GesturePhase.csv', dataset_name=DatasetHandler.GESTURE_PHASE)
    create_dataset_partitions('PATH_TO_DATA/data/OpenML/Gas/Gas.csv', dataset_name=DatasetHandler.GAS)
    create_dataset_partitions('PATH_TO_DATA/data/OpenML/House/house.csv', dataset_name=DatasetHandler.HOUSE)
    create_dataset_partitions('PATH_TO_DATA/data/OpenML/RobotNavigation/RobotNavigation.csv', dataset_name=DatasetHandler.ROBOT_NAVIGATION)

    syn_names = ['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']
    d_arr = [11, 50, 100, 150, 200, 250, 300]
    for syn in syn_names:
        for d in d_arr:
            generate_data(d=d, data_type=syn, output_dir='FeatureSelectionSynExp')