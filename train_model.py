''' Script to train CLAD model
'''

import torch as T
import argparse
from utils.configs import compose_config, load_yaml_config, resolve_config


from model.model import ContrastiveMLP
from losses.clad_loss import CLADLoss
import time


from data.load_data import get_data
from data.loaders import tabular_dl
from util.schedules import WarmupCosineSchedule, LRSchedule
import sys
from util.meter import AverageMeter
from util.checkpoint import make_checkpoint

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # data config
    parser.add_argument('--dataset_path', type=str, default='configs/datasets/lycos2017.yaml', help='path to dataset config')
    
    # model config
    parser.add_argument('--model_config_path', type=str, default='flow_id,src_addr,src_port,dst_addr,dst_port,ip_prot,timestamp', help='columns to drop from dataset')
    
    # loss config
    parser.add_argument('--loss_config_path', type=str, default='flow_id,src_addr,src_port,dst_addr,dst_port,ip_prot,timestamp', help='columns to drop from dataset')
    
    # hyperparameter config
    parser.add_argument('--hyperparameter_config_path', type=str, default='flow_id,src_addr,src_port,dst_addr,dst_port,ip_prot,timestamp', help='columns to drop from dataset')
    
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

def set_loader(opt):
    x_train, y_train, _, _, _, _, _, _ = get_data(
        data_path = opt.data_path, 
        target = 'label', 
        drop = opt.drop_cols, 
        class_zero = 'benign', 
        sample_thres = opt.sample_thres,
        split_seed = opt.split_seed,
        test_ratio = 0.5,
        val_ratio = 0.0,
    )
    
    train_dl = tabular_dl(
        x = x_train,
        y = y_train,
        batch_size = opt.batch_size, 
        balanced = True,
        collate_fn = None,
        drop_last = True,
        num_workers = 0,
    )
    
    return train_dl

def set_model(opt):
    # get model
    model = ContrastiveMLP(
        d_in = 72,
        n_classes = opt.n_classes,
        d_out = opt.d_out,
        neurons = opt.neurons,
        dropout = opt.dropout,
        residual = opt.residual,
    )
    model = model.to(opt.device)
    
    # get loss
    criterion = CLADLoss(
        m = opt.margin,
        squared = opt.squared,
    )
    return model, criterion

def set_optimiser(opt, model, train_dl):
    optimiser = T.optim.AdamW(
        model.parameters(),
        lr=1e-6, # initial learning rate
        betas = (0.9, 0.999),
        weight_decay = opt.weight_decay,
    )
    
    base_schedule = WarmupCosineSchedule(
        start_val = 1e-6,
        end_val = 1e-6,
        ref_val = opt.lr,
        T_max = (opt.epochs * len(train_dl)),
        warmup_steps =  int((opt.epochs//10) * len(train_dl)),
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
        loss = criterion(
            x = z,
            y = y,
        )

        # update metric
        losses.update(loss.item(), bsz)
        frac_pos_vals.update(criterion.get_fraction_pos())
        
        # optimser
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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

    # get config files
    data_config = compose_config(opt.dataset_path)
    hyperparameter_config = compose_config(opt.dataset_path)
    model_config = compose_config(opt.dataset_path)
    loss_config = compose_config(opt.dataset_path)

    dataset_cfg = compose_config(opt.dataset_path)
    model_cfg = load_yaml_config(opt.model_config_path)
    
    model_cfg["dataset"] = dataset_cfg
    model_cfg = resolve_config(model_cfg)

    
    # build data loader
    train_loader = set_loader(opt)
    
    # build model and criterion
    model, criterion = set_model(opt)
    
    # build optimizer
    optimiser, schedule = set_optimiser(opt, model, train_loader)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimiser, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        schedule.step()
        
    # save the trained model
    make_checkpoint(
        model = model, 
        optimiser = optimiser, 
        schedular = schedule, 
        path = 'weights/clad.pt.tar', 
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
