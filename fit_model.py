"""Hyperparameter search for triplet network"""

import torch as T
import argparse
import time
import os
import sys
import multiprocessing
from multiprocessing import Manager
from functools import partial
import traceback
import json
import pandas as pd

from data.load_data import get_data
from data.loaders import tabular_dl
from data.limited_data import get_limited_train_set, cv_generator
from triplet_network.model import ContrastiveMLP
from triplet_network.loss import BatchAllTripletLoss
from utils.schedules import CosineSchedule, LRSchedule
from utils.meter import AverageMeter
from utils.write_csv import append_csv, NoContext
from utils.configs import parse_config, compose_all_configs
from utils.process_batch import extract_features
from triplet_network.knn_classifier import knn_classifier
from utils.model_eval import model_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run config-driven triplet hyperparameter search."
    )

    # training sample size
    parser.add_argument(
        "--n_malicious",
        type=int,
        default=160,
        help="number of malicious samples per class",
    )
    parser.add_argument(
        "--n_benign", type=int, default=10000, help="number of benign samples"
    )

    # data config
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="configs/datasets/lycos2017.yaml",
        help="path to dataset config",
    )

    # model config
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="configs/hyperparameter_search/rand_model.yaml",
        help="path to model config",
    )

    # loss config
    parser.add_argument(
        "--loss_config_path",
        type=str,
        default="configs/hyperparameter_search/rand_loss.yaml",
        help="path to loss config",
    )

    # hyperparameter config
    parser.add_argument(
        "--hyperparameter_config_path",
        type=str,
        default="configs/hyperparameter_search/rand_hyperparameters.yaml",
        help="path to hyperparameter config",
    )

    # seeds
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=19048,
        help="sample seed for limited data sampling",
    )
    parser.add_argument(
        "--hpo_seed",
        type=int,
        default=4564,
        help="sample seed for limited data sampling",
    )
    parser.add_argument(
        "--cv_seed",
        type=int,
        default=19324,
        help="sample seed for limited data sampling",
    )

    parser.add_argument(
        "--n_cv_folds", type=int, default=5, help="number of cross-validation folds"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=10, help="repeat procedure this many times"
    )
    parser.add_argument(
        "--n_hpo_iterations",
        type=int,
        default=200,
        help="number of random search iterations",
    )
    parser.add_argument("--n_workers", type=int, default=1, help="number of workers")
    parser.add_argument(
        "--results_path",
        type=str,
        default="results/hp_search/search.csv",
        help="path to results file",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--print_freq", type=int, default=100, help="how many batches to print after"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="chunk size feature extraction and KNN classifier during inference",
    )

    opt = parser.parse_args()

    # add .yaml to config paths if not present
    if not opt.dataset_path.endswith(".yaml"):
        opt.dataset_path += ".yaml"

    if not opt.model_config_path.endswith(".yaml"):
        opt.model_config_path += ".yaml"

    if not opt.loss_config_path.endswith(".yaml"):
        opt.loss_config_path += ".yaml"

    if not opt.hyperparameter_config_path.endswith(".yaml"):
        opt.hyperparameter_config_path += ".yaml"

    if opt.results_path:
        if not opt.results_path.endswith(".csv"):
            opt.results_path += ".csv"

    if opt.results_path == "" or opt.results_path == " ":
        opt.results_path = None

    return opt


def set_model(cnf, opt):
    # get model
    model = ContrastiveMLP(
        d_in=cnf["model"]["d_in"],
        n_classes=cnf["model"]["n_classes"],
        d_out=cnf["model"]["d_out"],
        neurons=cnf["model"]["neurons"],
        dropout=cnf["model"]["dropout"],
    )
    model = model.to(opt.device)

    # get loss
    criterion = BatchAllTripletLoss(
        m=cnf["loss"]["m"],
        squared=cnf["loss"]["squared"],
    )

    return model, criterion


def set_optimiser(cnf, model, train_dl):
    optimiser = T.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        weight_decay=cnf["hyperparameters"]["optimiser"]["weight_decay"],
    )

    base_schedule = CosineSchedule(
        start_val=cnf["hyperparameters"]["schedules"]["lr"]["start_val"],
        end_val=cnf["hyperparameters"]["schedules"]["lr"]["end_val"],
        T_max=(cnf["hyperparameters"]["training_epochs"] * len(train_dl)),
    )

    lr_schedule = LRSchedule(
        optimiser,
        schedule=base_schedule,
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
        loss = criterion(z, y)

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
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "frac_pos {frac_pos_vals.val:.3f} ({frac_pos_vals.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    frac_pos_vals=frac_pos_vals,
                )
            )
            sys.stdout.flush()

    return losses.avg


def run_trial(
    x_data,
    y_data,
    cnf: dict,  # config with vairables unresolved
    opt,
    hpo_iteration: int,  # hyperparameter search iteration number
    run_iteration: int,  # run number within the current hyperparameter search iteration
    cv_fold_num: int,  # current cv fold
    lock=NoContext(),
    config_id=None,  # optional id for filtering results
):
    # resolve config variables based on current hpo and run iteration
    cnf = parse_config(
        cnf,
        seed=opt.hpo_seed + hpo_iteration + (opt.n_hpo_iterations * run_iteration),
    )

    # split data based on current run iteration and cv fold
    x_train, y_train, _, _ = get_limited_train_set(
        x_data=x_data,
        y_data=y_data,
        benign_samples=opt.n_benign,
        attack_samples=opt.n_malicious,
        seed=opt.subset_seed + run_iteration,
        set_num=0,
        replacement=False,
    )

    x_train, y_train, x_val, y_val = cv_generator(
        x_data=x_train,
        y_data=y_train,
        fold=cv_fold_num,
        n_folds=opt.n_cv_folds,
        seed=opt.cv_seed,
    )

    # renormalise data
    x_val = (x_val - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

    # build dataloader
    train_dl = tabular_dl(
        x=x_train,
        y=y_train,
        batch_size=cnf["hyperparameters"]["batch_size"],
        balanced=True,
        collate_fn=None,
        drop_last=True,
        num_workers=0,
    )

    # get model and optimiser based on config
    model, criterion = set_model(cnf, opt)
    optimiser, lr_schedule = set_optimiser(cnf, model, train_dl)

    # train model
    for epoch in range(1, cnf["hyperparameters"]["training_epochs"] + 1):
        _ = train(train_dl, model, criterion, optimiser, lr_schedule, epoch, opt)

    # get train and val embeddings
    x_train_t = T.tensor(x_train, dtype=T.float32, device=opt.device)
    y_train_t = T.tensor(y_train, dtype=T.int64, device=opt.device)
    x_val_t = T.tensor(x_val, dtype=T.float32, device=opt.device)
    y_val_t = T.tensor(y_val, dtype=T.int64, device=opt.device)

    train_embeddings, train_labels = extract_features(
        model=model,
        x_data=x_train_t,
        y_data=y_train_t,
        batch=True,
        chunk_size=opt.chunk_size,
    )

    val_embeddings, val_labels = extract_features(
        model=model,
        x_data=x_val_t,
        y_data=y_val_t,
        batch=True,
        chunk_size=opt.chunk_size,
    )

    # get model performance for each k value on validation set
    results = []

    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        y_true, y_pred = knn_classifier(
            x_train=train_embeddings,
            y_train=train_labels,
            x_test=val_embeddings,
            y_test=val_labels,
            k=k,
            dist_fn="euclidean",
            weight_fn="hard",
            num_chunks=max(
                1, (val_embeddings.size(0) + opt.chunk_size - 1) // opt.chunk_size
            ),
            num_classes=int(train_labels.max().item()) + 1,
        )

        k_results = model_eval(
            y_true=y_true,
            y_pred=y_pred,
            label="val",
            return_class_level=True,
            return_detection_metrics=True,
        )
        k_results["k"] = k

        k_results = {
            "config_id": config_id,
            "hpo_iteration": hpo_iteration,
            "run_iteration": run_iteration,
            "cv_fold_num": cv_fold_num,
            "config_seed": opt.hpo_seed
            + hpo_iteration
            + (opt.n_hpo_iterations * run_iteration),
            "subset_seed": opt.subset_seed + run_iteration,
            "cv_seed": opt.cv_seed,
            "config": json.dumps(cnf),
            "opt": json.dumps(vars(opt)),
            **k_results,
        }

        results.append(k_results)

    # write to csv
    if opt.results_path:
        with lock:
            append_csv(
                data=results,
                path=opt.results_path,
                quick_add=False,
            )

    return results  # return perfomance as list of dicts, one per k value


def run_trial_from_dict(trial, *, x_data, y_data, cnf, opt, lock):
    return run_trial(
        x_data=x_data,
        y_data=y_data,
        cnf=cnf,
        opt=opt,
        lock=lock,
        **trial,
    )


def main() -> None:

    # get args
    opt = parse_args()

    # Load and resolve all config files together so cross-config references work.
    cnf = compose_all_configs(
        {
            "dataset": opt.dataset_path,
            "hyperparameters": opt.hyperparameter_config_path,
            "model": opt.model_config_path,
            "loss": opt.loss_config_path,
        }
    )

    x_train, y_train, _, _, _, _, _, _ = get_data(
        data_path=cnf["dataset"]["data_path"],
        target="label",
        drop=cnf["dataset"]["drop_cols"],
        class_zero="benign",
        sample_thres=cnf["dataset"]["sample_thres"],
        split_seed=cnf["dataset"]["split_seed"],
        test_ratio=0.5,
        val_ratio=0.0,
    )

    # init results dir
    csv_dir = os.path.dirname(opt.results_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    trials = []

    for run_iteration in range(opt.n_iterations):
        for hpo_iteration in range(opt.n_hpo_iterations):
            for cv_fold_num in range(opt.n_cv_folds):
                config_id = (
                    run_iteration * opt.n_hpo_iterations * opt.n_cv_folds
                    + hpo_iteration * opt.n_cv_folds
                    + cv_fold_num
                )

                trials.append(
                    {
                        "config_id": config_id,
                        "hpo_iteration": hpo_iteration,
                        "run_iteration": run_iteration,
                        "cv_fold_num": cv_fold_num,
                    }
                )

    # skip already completed trials if results file exists
    total_trials = len(trials)
    completed_ids = set()

    if opt.results_path and os.path.exists(opt.results_path):
        completed_ids = set(pd.read_csv(opt.results_path)["config_id"].astype(int))

    trials = [trial for trial in trials if trial["config_id"] not in completed_ids]

    print(f"Number of total trials: {total_trials}.")
    print(f"Number of completed trials: {len(completed_ids)}.")
    print(f"Number of remaining trials: {len(trials)}.")

    # run experiments in parallel with multiprocessing
    if opt.n_workers > 1:
        p = None
        try:
            with Manager() as manager:
                lock = manager.Lock()
                worker_fn = partial(
                    run_trial_from_dict,
                    x_data=x_train,
                    y_data=y_train,
                    cnf=cnf,
                    opt=opt,
                    lock=lock,
                )

                with multiprocessing.Pool(opt.n_workers) as p:
                    p.map(worker_fn, trials)

        except Exception as e:
            print(e)
            traceback.print_exc()
            sys.stdout.flush()
            raise RuntimeError("ERROR CAUGHT STOPPING CODE")

        finally:
            if p is not None:
                p.close()
                p.join()

    else:
        for kwa in trials:
            run_trial(
                x_data=x_train,
                y_data=y_train,
                cnf=cnf,
                opt=opt,
                hpo_iteration=kwa["hpo_iteration"],
                run_iteration=kwa["run_iteration"],
                cv_fold_num=kwa["cv_fold_num"],
                config_id=kwa["config_id"],
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
