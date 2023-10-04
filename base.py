
import torch
import logging
from logging import getLogger
import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.model.general_recommender import *
import scipy.sparse as sp

from BaseCPALGC import BaseCPALGC


def main(exp_num, n_layer, learning_rate, reg_weight):
    dataset_name = f'YMB{exp_num}'
    ncri_table = {'TA5': 8, 'YM5': 5, 'RB5': 5, 'RA5': 5, 'YP5': 4}
    n_cri = 5  # ncri_table[dataset_name]
    epoch = 150
    parameter_dict = {
        'gpu_id': '0',
        'benchmark_filename': ['tr', 'val', 'ts'],
        'data_path': '',
        'seed': 3,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'user_inter_num_interval': "[0,inf)",
        'item_inter_num_interval': "[0,inf)",
        'load_col': {'inter': ['user_id', 'item_id', 'rating']},
        'neg_sampling': None,
        'epochs': epoch,
        'metrics': ['Precision', 'Recall', 'NDCG'],
        'topk': [5, 10],
        'device': torch.device('cuda'),
        'embedding_size': 64,
        'n_layers': n_layer,  # 2
        'learning_rate': learning_rate,  # 1e-3
        'reg_weight': reg_weight,  # 1e-2
        'eval_args': {'split': {'RS': [0.7, 0.15, 0.15]}}
    }

    config = Config(model='LightGCN', dataset=dataset_name,
                    config_dict=parameter_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    # Dataset preparation
    datasets = create_dataset(config)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, datasets)

    # Model and learning,
    # model (CPA-LGC) loading and initialization
    model = BaseCPALGC(config, train_data.dataset, n_cri).to(config['device'])

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data)

    # Evaluation
    results = trainer.evaluate(valid_data)
    # results = trainer.evaluate(test_data)
    logger.info(results)

    print(results)

    return str(results)


if __name__ == '__main__':
    n_layers = [1, 2, 3, 4, 5]
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    reg_weights = [1e-5, 1e-4, 1e-3, 1e-2]

    with open("base.txt", "a") as file:
        for n_layer in n_layers:
            for learning_rate in learning_rates:
                for reg_weight in reg_weights:
                    file.write('\n\n' + str(n_layer) + ' : ' +
                               str(learning_rate) + ' : ' + str(reg_weight) + '\n')

                    for i in range(1, 4):
                        result = main(i, n_layer, learning_rate, reg_weight)

                        file.write(result + '\n')

                    file.flush()
