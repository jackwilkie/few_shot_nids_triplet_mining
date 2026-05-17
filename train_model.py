''' Script to train CLAD model
'''

import torch as T
import argparse
import time
import os
import sys

from utils.configs import compose_all_configs
from utils.schedules import CosineSchedule, LRSchedule
from utils.meter import AverageMeter
from utils.checkpoint import make_checkpoint
from data.load_data import get_data
from data.loaders import tabular_dl
from data.limited_data import get_limited_train_set
from triplet_network.model import ContrastiveMLP
from triplet_network.loss import BatchAllTripletLoss

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
    
    parser.add_argument('--checkpoint_path', type=str, default='results/triplet_network/triplet_network.pt.tar', help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--print_freq', type=int, default=100, help='how many batches to print after')
    
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

def set_loader(cnf):
    
    # load dataset
    x_train, y_train, _, _, _, _, _, _ = get_data(
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
    x_train = (x_train - x_train.mean(axis = 0)) / x_train.std(axis = 0, ddof = 1)
    
    # build data loader
    train_dl = tabular_dl(
        x = x_train,
        y = y_train,
        batch_size = cnf['hyperparameters']['batch_size'], 
        balanced = True,
        collate_fn = None,
        drop_last = True,
        num_workers = 0,
    )
    
    return train_dl

def set_model(cnf):
    # get model
    model = ContrastiveMLP(
        d_in = cnf['model']['d_in'],
        n_classes = cnf['model']['n_classes'],
        d_out = cnf['model']['d_out'],
        neurons = cnf['model']['neurons'],
        dropout = cnf['model']['dropout'],
    )
    model = model.to(cnf['device'])
    
    # get loss
    criterion = BatchAllTripletLoss(
        m = cnf['loss']['m'],
        squared = cnf['loss']['squared'],
    )
    
    return model, criterion

def set_optimiser(cnf, model, train_dl):
    optimiser = T.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        betas = (0.9, 0.999),
        weight_decay = cnf['hyperparameters']['optimiser']['weight_decay'],
    )
    
    base_schedule = CosineSchedule(
        start_val = cnf['hyperparameters']['schedules']['lr']['start_val'],
        end_val = cnf['hyperparameters']['schedules']['lr']['end_val'],
        T_max = (cnf['hyperparameters']['training_epochs'] * len(train_dl)),
    )
    
    lr_schedule = LRSchedule(
        optimiser,
        schedule = base_schedule,
    )
    
    return optimiser, lr_schedule

def train(
    train_loader, 
    model, 
    criterion, 
    optimizer, 
    schedule,
    epoch, 
    opt,
):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    frac_pos_vals = AverageMeter()

    end = time.time()
    for idx, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        x = x.to(opt.device)
        y = y.to(opt.device)
        bsz = y.size(0)

        z = model(x)        
        loss = criterion(z,y)

        # update metric
        losses.update(loss.item(), bsz)
        frac_pos_vals.update(criterion.get_fraction_pos())
        
        # optimser
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'frac_pos {frac_pos_vals.val:.3f} ({frac_pos_vals.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, frac_pos_vals = frac_pos_vals))
            sys.stdout.flush()

    return losses.avg


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
    
    # build data loader
    train_loader = set_loader(cnf)
    
    # build model and criterion
    model, criterion = set_model(cnf)
    
    # build optimizer
    optimiser, schedule = set_optimiser(cnf, model, train_loader)

    # training routine
    for epoch in range(1, cnf['hyperparameters']['training_epochs'] + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimiser, schedule, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

    # save the trained model
    os.makedirs(os.path.dirname(opt.checkpoint_path), exist_ok=True)
    make_checkpoint(
        model = model, 
        optimiser = optimiser, 
        schedular = schedule, 
        path = opt.checkpoint_path, 
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
