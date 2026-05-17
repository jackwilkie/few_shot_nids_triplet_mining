"""Script to evaluate triplet network from saved weights"""

import torch as T
import argparse
from pprint import pprint
import json
import pandas as pd
import os
import numpy as np
import time
import sys

from data.load_data import get_data
from utils.process_batch import extract_features
from data.limited_data import get_limited_train_set
from triplet_network.model import ContrastiveMLP
from triplet_network.knn_classifier import knn_classifier
from utils.model_eval import model_eval
from data.loaders import tabular_dl
from utils.schedules import CosineSchedule, LRSchedule
from utils.meter import AverageMeter
from triplet_network.loss import BatchAllTripletLoss


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

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

    parser.add_argument(
        "--n_iterations", type=int, default=10, help="training iteration"
    )

    parser.add_argument(
        "--hpo_path",
        type=str,
        default="results/hp_search/search.csv",
        help="path to hyperparameter search results file",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="results/hp_search/best_performance.json",
        help="path to results file",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="chunk size feature extraction and KNN classifier during inference",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=19048,
        help="sample seed for limited data sampling",
    )
    parser.add_argument(
        "--print_freq", type=int, default=100, help="how many batches to print after"
    )
    opt = parser.parse_args()

    if not opt.hpo_path.endswith(".csv"):
        opt.hpo_path += ".csv"

    if not opt.results_path.endswith(".json"):
        opt.results_path += ".json"

    return opt


def load_data(cnf):

    # load dataset
    x_train, y_train, _, _, x_test, y_test, _, _ = get_data(
        data_path=cnf["dataset"]["data_path"],
        target="label",
        drop=cnf["dataset"]["drop_cols"],
        class_zero="benign",
        sample_thres=cnf["dataset"]["sample_thres"],
        split_seed=cnf["dataset"]["split_seed"],
        test_ratio=0.5,
        val_ratio=0.0,
    )

    # sample limited training set from training data
    x_train, y_train, _, _ = get_limited_train_set(
        x_data=x_train,
        y_data=y_train,
        benign_samples=cnf["dataset"]["n_benign"],
        attack_samples=cnf["dataset"]["n_malicious"],
        seed=cnf["dataset"]["limited_sample_seed"],
        set_num=cnf["dataset"]["iteration"],
        replacement=False,
    )

    # renormalise data
    x_test = (x_test - x_train.mean(axis=0)) / x_test.std(axis=0, ddof=1)
    x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

    # convert to tensore and return
    x_train = T.tensor(x_train, dtype=T.float32, device=cnf["device"])
    y_train = T.tensor(y_train, dtype=T.int64, device=cnf["device"])
    x_test = T.tensor(x_test, dtype=T.float32, device=cnf["device"])
    y_test = T.tensor(y_test, dtype=T.int64, device=cnf["device"])

    return x_train, y_train, x_test, y_test


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


def triplet_network_eval(
    x_train,
    y_train,
    x_test,
    y_test,
    cnf,
    best_k: int,
    iteration,
    opt,
) -> dict:

    # -- train model

    # get train data
    x_train, y_train, _, _ = get_limited_train_set(
        x_data=x_train,
        y_data=y_train,
        benign_samples=opt.n_benign,
        attack_samples=opt.n_malicious,
        seed=opt.subset_seed + iteration,
        set_num=0,
        replacement=False,
    )

    # renormalise data
    x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
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

    model.eval()

    # -- eval model
    with T.no_grad():
        x_train_t = T.tensor(x_train, dtype=T.float32, device=opt.device)
        y_train_t = T.tensor(y_train, dtype=T.int64, device=opt.device)
        x_test_t = T.tensor(x_test, dtype=T.float32, device=opt.device)
        y_test_t = T.tensor(y_test, dtype=T.int64, device=opt.device)

        train_embeddings, train_labels = extract_features(
            model=model,
            x_data=x_train_t,
            y_data=y_train_t,
            batch=True,
            chunk_size=opt.chunk_size,
        )

        test_embeddings, test_labels = extract_features(
            model=model,
            x_data=x_test_t,
            y_data=y_test_t,
            batch=True,
            chunk_size=opt.chunk_size,
        )

        y_true, y_pred = knn_classifier(
            x_train=train_embeddings,
            y_train=train_labels,
            x_test=test_embeddings,
            y_test=test_labels,
            k=best_k,
            dist_fn="euclidean",
            weight_fn="hard",
            num_chunks=max(
                1, (test_embeddings.size(0) + opt.chunk_size - 1) // opt.chunk_size
            ),
            num_classes=int(train_labels.max().item()) + 1,
        )

    return model_eval(
        y_true=y_true,
        y_pred=y_pred,
        label="test",
        return_class_level=True,
        return_detection_metrics=True,
    )


def main():
    opt = parse_option()

    # load random search results
    df = pd.read_csv(opt.hpo_path)

    # get best configs
    best_results = []

    for run_iteration in range(opt.n_iterations):
        run_df = df[df["run_iteration"] == run_iteration]

        # skip iterations not yet present in the CSV
        if run_df.empty:
            continue

        # mean across CV folds for each sampled config + k pair
        mean_scores = run_df.groupby(["hpo_iteration", "k"], as_index=False)[
            "val_Macro_F1"
        ].mean()

        # best config/k pair for this run iteration
        best_row = mean_scores.loc[mean_scores["val_Macro_F1"].idxmax()]

        best_hpo_iteration = int(best_row["hpo_iteration"])
        best_k = int(best_row["k"])

        # get the actual stored config for that best pair
        matching_rows = run_df[
            (run_df["hpo_iteration"] == best_hpo_iteration) & (run_df["k"] == best_k)
        ]

        first_row = matching_rows.iloc[0]

        best_results.append(
            {
                "run_iteration": run_iteration,
                "hpo_iteration": best_hpo_iteration,
                "k": best_k,
                "mean_val_Macro_F1": float(best_row["val_Macro_F1"]),
                "config": json.loads(first_row["config"]),
            }
        )

    # -- get config for iteration 0 (can be any iteration since they use same train/test splits)
    cnf = best_results[0]["config"]
    x_train, y_train, _, _, x_test, y_test, _, _ = get_data(
        data_path=cnf["dataset"]["data_path"],
        target="label",
        drop=cnf["dataset"]["drop_cols"],
        class_zero="benign",
        sample_thres=cnf["dataset"]["sample_thres"],
        split_seed=cnf["dataset"]["split_seed"],
        test_ratio=0.5,
        val_ratio=0.0,
    )

    # eval each available iteration
    results = [
        triplet_network_eval(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cnf=result["config"],
            best_k=result["k"],
            iteration=result["run_iteration"],
            opt=opt,
        )
        for result in best_results
    ]

    # mean results dicts
    mean_results = {
        key: np.mean([result[key] for result in results]) for key in results[0]
    }

    # print metrics and save as json
    pprint(mean_results)

    os.makedirs(os.path.dirname(opt.results_path), exist_ok=True)

    with open(opt.results_path, "w") as f:
        json.dump(mean_results, f, indent=4)


if __name__ == "__main__":
    main()
