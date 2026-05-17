''' Script to evaluate triplet network from saved weights
'''

import torch as T
import torch.nn as nn
from torch import Tensor
import argparse
from pprint import pprint
import json
import os

from data.load_data import get_data
from utils.checkpoint import load_checkpoint
from utils.process_batch import extract_features
from utils.configs import compose_all_configs
from data.limited_data import get_limited_train_set
from triplet_network.model import ContrastiveMLP
from triplet_network.knn_classifier import knn_classifier
from utils.model_eval import model_eval


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training sample size
    parser.add_argument('--n_malicious', type=int, default=160, help='number of malicious samples per class')
    parser.add_argument('--n_benign', type=int, default=10000, help='number of benign samples')
    parser.add_argument('--limited_sample_seed', type=int, default=19048, help='sample seed for limited data sampling')
    parser.add_argument('--iteration', type=int, default=0, help='training iteration')
    
    # data config
    parser.add_argument('--dataset_path', type=str, default='configs/datasets/lycos2017.yaml', help='path to dataset config')
    
    # model config
    parser.add_argument('--model_config_path', type=str, default='configs/model.yaml', help='path to model config')
    
    # loss config
    parser.add_argument('--loss_config_path', type=str, default='configs/loss.yaml', help='path to loss config')
    
    # hyperparameter config
    parser.add_argument('--hyperparameter_config_path', type=str, default='configs/hyperparameters.yaml', help='path to hyperparameter config')
    parser.add_argument('--k', type=int, default=8, help='k value for KNN classifier during inference')
    
    parser.add_argument('--checkpoint_path', type=str, default='results/triplet_network/triplet_network.pt.tar', help='path to checkpoint file')
    parser.add_argument('--results_path', type=str, default='results/triplet_network/performance.json', help='path to results file')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--print_freq', type=int, default=100, help='how many batches to print after')
    parser.add_argument('--chunk_size', type=int, default=1024, help='chunk size feature extraction and KNN classifier during inference')
    
    opt = parser.parse_args()
    
    # add .yaml to config paths if not present
    if not opt.dataset_path.endswith('.yaml'):
        opt.dataset_path += '.yaml'
        
    if not opt.model_config_path.endswith('.yaml'):
        opt.model_config_path += '.yaml'
    
    if not opt.loss_config_path.endswith('.yaml'):
        opt.loss_config_path += '.yaml'
        
    if not opt.hyperparameter_config_path.endswith('.yaml'):
        opt.hyperparameter_config_path += '.yaml'
    
    return opt


def load_data(cnf):
    
    # load dataset
    x_train, y_train, _, _, x_test, y_test, _, _ = get_data(
        data_path = cnf['dataset']['data_path'], 
        target = 'label', 
        drop = cnf['dataset']['drop_cols'], 
        class_zero = 'benign', 
        sample_thres = cnf['dataset']['sample_thres'],
        split_seed = cnf['dataset']['split_seed'],
        test_ratio = 0.5,
        val_ratio = 0.0,
    )
    
    # sample limited training set from training data
    x_train, y_train, _, _ = get_limited_train_set(
        x_data = x_train,
        y_data = y_train, 
        benign_samples = cnf['dataset']['n_benign'], 
        attack_samples = cnf['dataset']['n_malicious'], 
        seed = cnf['dataset']['limited_sample_seed'], 
        set_num= cnf['dataset']['iteration'], 
        replacement = False,
    )
    
    # renormalise data
    x_test = (x_test - x_train.mean(axis = 0)) / x_test.std(axis = 0, ddof = 1)
    x_train = (x_train - x_train.mean(axis = 0)) / x_train.std(axis = 0, ddof = 1)
    
    # convert to tensore and return 
    x_train = T.tensor(x_train, dtype = T.float32, device = cnf['device'])
    y_train = T.tensor(y_train, dtype = T.int64, device = cnf['device'])
    x_test = T.tensor(x_test, dtype = T.float32, device = cnf['device'])
    y_test = T.tensor(y_test, dtype = T.int64, device = cnf['device'])

    return x_train, y_train, x_test, y_test

def load_model(cnf, opt):
    # get model
    model = ContrastiveMLP(
        d_in = cnf['model']['d_in'],
        n_classes = cnf['model']['n_classes'],
        d_out = cnf['model']['d_out'],
        neurons = cnf['model']['neurons'],
        dropout = cnf['model']['dropout'],
    )
    model = model.to(opt.device)
    
    # load weights
    model, _, _, _, _ = load_checkpoint(
        opt.checkpoint_path,
        model,
    )
    
    model = model.to(opt.device)
    model.eval()
    return model


@T.no_grad()
def triplet_network_eval(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    opt,
) -> dict:

    # extract embeddings from training data
    train_embeddings, train_labels = extract_features(
        model=model,
        x_data=x_train,
        y_data=y_train,
        batch=True,
        chunk_size=opt.chunk_size
    )

    # extract embeddings from test data
    test_embeddings, test_labels = extract_features(
        model=model,
        x_data=x_test,
        y_data=y_test,
        batch=True,
        chunk_size=opt.chunk_size
    )
    
    y_true, y_pred = knn_classifier(
        x_train=train_embeddings,
        y_train=train_labels,
        x_test=test_embeddings,
        y_test=test_labels,
        k=opt.k,
        dist_fn='euclidean',
        weight_fn='hard',
        num_chunks=max(1, (test_embeddings.size(0) + opt.chunk_size - 1) // opt.chunk_size),
        num_classes=int(train_labels.max().item()) + 1,
    )

    return model_eval(
        y_true=y_true,
        y_pred=y_pred,
        label='test',
        return_class_level=True,
        return_detection_metrics=True,
    )

def main():
    opt = parse_option()

    # Load and resolve all config files together so cross-config references work.
    cnf = compose_all_configs(
        {
            "dataset": opt.dataset_path,
            "hyperparameters": opt.hyperparameter_config_path,
            "model": opt.model_config_path,
            "loss": opt.loss_config_path,
        }
    )

    # inject sample settings into config
    cnf['dataset']['n_malicious'] = opt.n_malicious
    cnf['dataset']['n_benign'] = opt.n_benign
    cnf['dataset']['limited_sample_seed'] = opt.limited_sample_seed
    cnf['dataset']['iteration'] = opt.iteration
    cnf['device'] = opt.device
    
    # get data
    x_train, y_train, x_test, y_test = load_data(cnf)
    
    # get model
    model = load_model(cnf, opt)
    
    # eval model
    metrics = triplet_network_eval(
        model = model,
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test,
        opt = opt,
    )
    
    # print metrics and save as json
    pprint(metrics)
    
    os.makedirs(os.path.dirname(opt.results_path), exist_ok=True)

    with open(opt.results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
if __name__ == '__main__':
    main()
