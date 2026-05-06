#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file for image base experiments

Created on Mon Aug 14 13:04:01 2023

@author: jack
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from loggers.logger import Logger
import os
from utils.gpu import get_gpu_num
from utils.get_id import get_id
from factory.dataloaders import dataloader_factory
import pandas as pd
import torch as T
import torch.multiprocessing as mp
from model_training import distributed_training
import copy
from utils.config_parser import parse_config
import torch as T
import multiprocessing
from multiprocessing import Manager
from utils.no_context import NoContext
from utils import csv
from engines.engine_registry import call_engine_function
import traceback
import sys
import copy
import numpy as np
import yaml

def train_eval_model(
    rank: int,  
    config: OmegaConf, 
    world_size: int = 0,
    lock = None,
    queue = None,
    iteration = None,
    parent_id = None,
    ref_df = None,
    preloaded_data = None,
) -> None:
    """Function to set up experiment logging, checpointing and running experiment from config

    Args:
        rank (int): Rank of process in distributed training, 0 if not using distributed training
        config (OmegaConf): Experiment config file
        world_size (int, optional): World_size when using distributed training (number of distributed processes). Defaults to 0.
        lock (_type_, optional): Lock when using multiprocessing to run experiments concurrently, prevent simultaneous read/write. Defaults to None.
        queue (_type_, optional): Queue to get device id when running multiple experiments concurrently. Defaults to None.
        iteration (_type_, optional): Iteration number when running multiple experiments concurrently, None if not using multiprocessing. Defaults to None.
        parent_id (_type_, optional): Run name without iteration num when running multiple experiments concurrently. Defaults to None.

    Raises:
        ValueError: Raises when using distributed training but config parsing seed is not specified, prevents different processses having different configs
        ValueError: Raises when checkpoint path specified does not exist
        ValueError: Raises when too many experiments exist with specified run name (max is 10,000)
    """
    print('======entering main========')
    config = copy.deepcopy(config)  # copy config to prevent overwrite

    # -- make config copies and resolve
    unparsed_conf = copy.deepcopy(config)
    use_conf = copy.deepcopy(config)
    
    # parse config
    config_seed = use_conf['config_seed'] if 'config_seed' in use_conf else 0

    use_conf = parse_config(
        use_conf,
        seed = config_seed
        )
    
    # -- get training options
    distributed = True if world_size > 0 else False  # use distributed trianing if world_size > 0
    rank = None if not distributed else rank  # get rank if distributed training
    distributed_training.setup(rank, world_size)  # setup distributed training if world_size > 0
    lock = lock or NoContext()  # use lock when provided (multi-process training)

    # rank and world size required for distributed datasets
    use_conf['dataset']['rank'] = rank
    use_conf['dataset']['world_size'] = world_size
    
    if config_seed is None and distributed:
        # seed must be same for distributed training for consistency between processes
        raise ValueError('Please specify config seed when using distributed training')
    
    # -- check for checkpoint
    use_checkpoint = False if config.get('checkpoint_epoch',0) == 0 else True
    run_name = use_conf['logging']['run_name']
    project_name = use_conf['logging']['project']

    if run_name is None or project_name is None:
            raise ValueError('Checkpoint requested but project or run name not specified')
    
    # -- generate run name and get checkpoint
    csv_path = os.path.expanduser(use_conf['logging']['csv_path'])
    new_run_name = run_name
    if not use_checkpoint and iteration is None:
        with lock:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                df = pd.DataFrame(columns = ['run_name'])

        invalid_names = df['run_name'].tolist() if 'run_name' in df else []
        
        # -- set project to default if not specified
        if project_name is None:
            # use default project if name not specified
            project_name = 'default'
        
        if run_name is None:
            # generate run name if none providede
            run_name = get_id()

        # check if run name exists and add indentifying number if so
        run_number = 0
        
        # run name cant exist in csv file or have a pre-existing directory
        if iteration is None:
            while new_run_name in invalid_names or os.path.isdir(os.path.expanduser(f'{use_conf["logging"]["save_path"]}/{project_name}/{new_run_name}')):
                run_number += 1
                new_run_name = f'{run_name}_{run_number}'
                if run_number >= 10000:
                    raise ValueError('Valid Run Name Not Found')
        
    run_name = new_run_name
    start_epoch = use_conf.get('checkpoint_epoch', 0)
    save_dir = os.path.expanduser(f'{use_conf["logging"]["save_path"]}/{project_name}/{run_name}')  # create dir to save results and checkpoints
    checkpoint_path = use_conf['checkpoint_path'] if use_checkpoint else None

    # -- begin csv and w and b logging
    if iteration is not None:
        save_dir += f'/iteration_{iteration}'  # track iteration when using multiprocessing

    with lock:
        if ((distributed and rank == 0) or not distributed) and use_conf['logging']['log']:

            # init w and b logging
            if not use_checkpoint:
                os.makedirs(save_dir)  # create new save dir if it does not already exist due to loading checkpoint
            
            stats = Logger(project = use_conf['logging']['project'], run_name = run_name, config = use_conf, tags = use_conf['logging']['tags']) 
                
            with open(f'{save_dir}/config.yaml', 'w') as f:
                # save parsed config as yaml file
                OmegaConf.save(use_conf, f)
            
            with open(f'{save_dir}/config_unparsed.yaml', 'w') as f:
                # save un parsed config as yaml file
                OmegaConf.save(unparsed_conf, f)
        else:
            # only main process uses w and b logging 
            stats = None
        
    # -- get device immediately before needed
    
    if queue is not None:
        device = queue.get()  # use next device in queue for multiprocessing training
    elif distributed:
        device = use_conf['device'][rank % len(use_conf['device'])]  # use gpu corresponding to rank for distributed training, assumes devices is list
    elif use_conf['device'] is None:
        device = get_gpu_num()  # use least utilised gpu if none specifieed
    elif isinstance(use_conf['device'], list):
        # FIXME use least utilised gpu in list in this case
        device = use_conf['device'][0]  # use first gpu if multiple specified and one needed, in future will use least utilised in list
    else:
        device = use_conf['device']  # one gpu specififed
    
    
    # -- get dataloaders
    use_conf['dataset']['device'] = device  # give access to device for tabular data, where it is preloaded to gpu
    train_dl, val_dl, test_dl = None, None, None
    x_train, x_val, x_test = None, None, None
    y_train, y_val, y_test = None, None, None
    
    with lock:
        if preloaded_data is None:
            train_dl, val_dl, test_dl = dataloader_factory(use_conf)
        
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = preloaded_data
            
            '''
            #x_train, y_train = x_train.clone().to(device), y_train.clone().to(device)
            x_train, y_train = x_train, y_train
            if x_val is not None:
                x_val, y_val = x_val, y_val
            
            if x_test is not None:
                x_test, y_test = x_test, y_test
            '''
            
    # -- run experiment engine
    # get engine
    engine_arg_dict = {
        'train_dl': train_dl,
        'val_dl': val_dl,
        'test_dl': test_dl,
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        'device': device,
        'logger': stats,
        'config': use_conf,
        'config_unparsed': unparsed_conf,
        'distributed': distributed,
        'rank': rank,
        'world_size': world_size,
        'lock': lock,
        'checkpoint_path': checkpoint_path,
        'queue': queue,
        'iteration': iteration,
        'parent_id': parent_id,
        'run_name': run_name,
        'save_dir': save_dir,
        'csv_path': csv_path,
        'ref_df': ref_df,
    }
    engine_conf = use_conf.pop('engine', 'train')
    if isinstance(engine_conf, dict):
        engine_name = engine_conf['name'] 
        engine_arg_dict = {**engine_arg_dict, **engine_conf}
    else:
        engine_name = engine_conf
    records = call_engine_function(engine_name, engine_arg_dict)
    
    # -- clean up training and log
    if distributed: distributed_training.cleanup()  # close distributed backend
    if rank is None or rank == 0:

        if queue is not None:
            # replace device in queue if using multi-processing
            queue.put(device)
            
        if stats is not None:
            # finish w and b logging
            stats.finish()

        # log final model stats in csv file
        if records is not None:
            with lock:
                csv.append_csv(records, csv_path)  # append results to csv
            

def worker(args):
    return train_eval_model(
        rank = 0,
        world_size = 0,
        config = args['config'],
        iteration= args['iteration'],
        queue = args['queue'],
        lock = args['lock'],
        parent_id=args['id'],
        ref_df = args['ref_df'],
        preloaded_data= args['preloaded_data'],
    )

@hydra.main(version_base = None, config_path = 'configs', config_name = 'config')
def main(config: DictConfig):
    # -- check for config overwrite
    use_conf = OmegaConf.to_container(config, resolve=True) # resolve inital config
    ovr = use_conf.get('config_ovr', None)
    ovr = f'{ovr}.yaml' if not ovr.endswith('.yaml') else ovr
    
    # -- load new config and resolve if overide used
    if ovr is not None:
        use_conf = hydra.compose(ovr)
        use_conf = OmegaConf.to_container(use_conf, resolve=True)
    
    device = None
    use_checkpoint = use_conf.get('checkpoint', None)
    if use_checkpoint is not None:
        log_ = use_conf['logging']['log']
        checkpoint_config_path = os.path.expanduser(use_checkpoint['path'])
        if not checkpoint_config_path.endswith('.yaml'):
            checkpoint_config_path += '.yaml' 
        
        checkpoint_epoch = use_checkpoint['epoch']
        device = use_conf['device']      
        with open(checkpoint_config_path, 'r') as file:
            use_conf = yaml.safe_load(file)
        
        checkpoint_dir = checkpoint_config_path[:checkpoint_config_path.rfind('/')]
        run_name = checkpoint_dir[checkpoint_dir.rfind('/')+1:]
        use_conf['logging']['run_name'] = run_name
        use_conf['checkpoint_path'] = f'{checkpoint_dir}/{run_name}_epoch_{checkpoint_epoch}.pt.tar'
        use_conf['checkpoint_epoch'] = checkpoint_epoch
        use_conf['logging']['log'] = log_
        use_conf['device'] = device
        
    multiprocessing.set_start_method('spawn')
    
    # -- initalise processes and run 
    device = use_conf['device'] if device is None else device
    distributed_training = use_conf.get('distributed_training', False) 
    testing_mode = use_conf.get('testing_mode', False)
   
    world_size = 1 if not isinstance(device, list) else len(device)  # get world size (number of devices for distributed training)
    iterations = use_conf['iterations']  # number of parallel iteration

    ref_df_path = use_conf.get('ref_df_path', None)
    ref_df = None if ref_df_path is None else pd.read_csv(os.path.expanduser(ref_df_path))
    
    print(f'--- Distributed Training: {distributed_training}, world size: {world_size}, devices: {device}')
    
    if world_size > 1 and distributed_training:
        try:
            # -- use distrubted training
            mp.spawn(
                train_eval_model,
                args = (use_conf, world_size),
                nprocs = world_size
                )
            
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise RuntimeError('ERROR CAUGHT STOPPING CODE')
            
    elif iterations != 0 and distributed_training:
        # cant run both multiple instances in parallel and use distributed training
        raise ValueError('Please select either distributed or multirun training, not both')
    
    elif iterations > 1:
        queue = multiprocessing.Manager().Queue()
        processes_per_gpu = use_conf['processes_per_gpu']
        num_processes = world_size * processes_per_gpu
        spawn_id = get_id()
        
        #iterate over each process on each gpu
        for device_id in range(world_size):  
            for _ in range(processes_per_gpu):  
                queue.put(use_conf['device'][device_id]) #add each gpu to queue for each process per gpu
    
        
        # run multiple times in parallel
        try:
            with Manager() as manager:
                lock = manager.Lock()
                use_conf['run_id'] = get_id()
                csv_path = os.path.expanduser(use_conf['logging']['csv_path'])
                df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
                completed_its = pd.read_csv(csv_path)['iteration'].values if df is not None else []
                n_completed = len(df['iteration'].value_counts()) if df is not None else 0
                
                if use_conf.pop('preload_dl', False):
                    data_conf_ = copy.deepcopy(use_conf)
                    data_conf_['device'] = 'cpu'
                    data_conf_['dataset']['device'] = 'cpu'
                    data_conf_['dataset']['return_data'] = True
                    preloaded_data = dataloader_factory(data_conf_)
                else:
                    preloaded_data = None
                    
                # Preload data, so it is not done individually for each worker
                print(f'{n_completed} already copmleted')
                args = [
                    {
                        'iteration': i, 
                        'lock': lock, 
                        'config': use_conf, 
                        'queue': queue, 
                        'id': spawn_id, 
                        'ref_df': ref_df,
                        'preloaded_data': preloaded_data,
                        } 
                    for i in range(iterations) if not i in completed_its]
                
                print(f'{len(args)} its remain')
                
                if testing_mode:
                    p = None
                    for i in args:
                        worker(i)
                else:
                    with multiprocessing.Pool(num_processes) as p:
                        p.map(worker, args)

                print('finished')
                
        except Exception as e:
            print(e)
            traceback.print_exc()
            sys.stdout.flush()
            raise RuntimeError('ERROR CAUGHT STOPPING CODE')

        finally:
            if p is not None:
                p.close()
                p.join()
    
    else:
        train_eval_model(0, use_conf)

if __name__ == '__main__':
    main()
