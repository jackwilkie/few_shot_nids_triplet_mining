"""
Standalone hyperparameter search runner for triplet experiments.

The script composes one experiment config, validates that it is a triplet
setup, then runs repeated trials using the repo's config parser and training
entry point. Model, loss, dataset, optimiser, and hyperparameter search
expressions therefore stay in the YAML configs rather than in this script.
"""

import argparse
import multiprocessing
import os
import sys
from functools import partial
from multiprocessing import Manager
from typing import Any
from data.load_data import get_data

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run config-driven triplet hyperparameter search.")
    
    # configs
    parser.add_argument(
        "--dataset_config",
        default="configs/triplet_mining_config.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--model_config",
        default="configs/triplet_mining_config.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--loss_config",
        default="configs/triplet_mining_config.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--hyperparameter_config",
        default="configs/triplet_mining_config.yaml",
        help="Path to the experiment YAML config.",
    )
    
    parser.add_argument("--iterations", type=int, default=None, help="Number of hyperparameter search iterations to run.")
    parser.add_argument("--runs", type=int, default=None, help="Number of runs in each hyperparameter search.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--devices", nargs="+", default=None, help="Devices to schedule trials onto.")
    parser.add_argument("--csv-path", default=None, help="Override the results CSV path.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for config sampling.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip iterations already in the CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the planned trials only.")
    return parser.parse_args()


def get_dataset(config: dict) -> Any:
    config['device'] = 'cpu'
    config['dataset']['return_data'] = True
    x_train, y_train, _, _, _, _ = dataloader_factory(config)
    return x_train, y_train

def main() -> None:
    args = parse_args()
    config = compose_config(args.config)
    validate_triplet_config(config)
    
    # -- get dataset
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
    
    
    print('='*100)
    print(args)
    print('-'*100)
    print(config)
    print('='*100)
    raise NotImplementedError("This script is a template for running hyperparameter searches. Fill in the worker function and run_trial function with your training code, then run this script with your config to execute the search.")
    
    opts = get_search_options(config, args)
    trials = remaining_trials(opts["csv_path"], opts["n_trials"], opts["skip_completed"])

    print(f"config: {args.config}")
    print(f"csv: {opts['csv_path']}")
    print(f"devices: {opts['devices']}")
    print(f"workers: {opts['n_workers']}")
    print(f"remaining trials: {len(trials)} / {opts['n_trials']}")

    if args.dry_run or not trials:
        return

    csv_dir = os.path.dirname(opts["csv_path"])
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    if opts["n_workers"] > 1 and len(trials) > 1:
        multiprocessing.set_start_method("spawn", force=True)
        with Manager() as manager:
            lock = manager.Lock()
            queue = manager.Queue()
            for i in range(opts["n_workers"]):
                queue.put(opts["devices"][i % len(opts["devices"])])

            chunks = [trials[i:: opts["n_workers"]] for i in range(opts["n_workers"])]
            chunks = [chunk for chunk in chunks if chunk]
            worker_fn = partial(
                worker,
                base_config=config,
                lock=lock,
                queue=queue,
                csv_path=opts["csv_path"],
                seed=opts["seed"],
            )
            with multiprocessing.Pool(len(chunks)) as pool:
                pool.map(worker_fn, chunks)
    else:
        worker(
            trials,
            base_config=config,
            lock=None,
            queue=None,
            csv_path=opts["csv_path"],
            seed=opts["seed"],
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
