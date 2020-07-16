from pathlib import Path
import torch

import data
import train
import util

import os

# hack for https://github.com/dmlc/xgboost/issues/1715
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main(args):
    # general
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        args.cuda = True
    else:
        device = torch.device("cpu")
        args.cuda = False

    if args.test_run:
        args.num_iterations = 1
        args.test_interval = 2
    util.print_args(args)

    # data
    (
        data_train,
        data_valid,
        data_test,
        target_train,
        target_valid,
        target_test,
    ) = data.load_binarized_omniglot_with_targets(location=args.data_location)
    id_offset_test = len(data_train) + len(data_valid)

    if args.small_dataset:
        assert args.dataset_size in [None, 500]
        args.dataset_size = 500
        data_train, target_train = data.split_data_by_target(
            data_train, target_train, num_data_per_target=10
        )
        data_test = data_train
        target_test = target_train
    elif args.dataset_size is not None:
        data_train, target_train = data.split_data_by_target(
            data_train, target_train, num_data_per_target=args.dataset_size // 50
        )
    args.num_rows, args.num_cols = data_train.shape[1:]
    batch_size = args.batch_size

    if args.condition_on_alphabet:
        data_loader = data.get_conditional_omniglot_data_loader(
            data_train, target_train, batch_size, device, ids=True
        )
        test_data_loader = data.get_conditional_omniglot_data_loader(
            data_test, target_test, args.test_batch_size, device, id_offset=id_offset_test, ids=True
        )
    else:
        data_loader = data.get_data_loader(data_train, batch_size, device, ids=True)
        test_data_loader = data.get_data_loader(
            data_test, args.test_batch_size, device, id_offset=id_offset_test, ids=True
        )
    args.num_train_data = len(data_loader.dataset)

    # init
    checkpoint_path = util.get_checkpoint_path(args)
    if not Path(checkpoint_path).exists():
        util.logging.info("Training from scratch")
        (generative_model, inference_network, optimizer, memory, stats) = util.init(args, device)
    else:
        (
            generative_model,
            inference_network,
            optimizer,
            memory,
            stats,
            run_args,
        ) = util.load_checkpoint(checkpoint_path, device)

    # train
    if not Path(checkpoint_path).is_file():
        train.train_sleep(
            generative_model,
            inference_network,
            min(500, batch_size * args.num_particles),
            args.pretrain_iterations,
            args.log_interval,
        )
    algorithm_params = {
        "test_num_particles": args.test_num_particles,
        "num_iterations": args.num_iterations,
        "log_interval": args.log_interval,
        "test_interval": args.test_interval,
        "save_interval": args.save_interval,
        "checkpoint_path": checkpoint_path,
    }
    if args.algorithm == "rws" or args.algorithm == "vimco":
        algorithm_params = {
            **algorithm_params,
            "num_particles": args.num_particles,
        }
    elif args.algorithm == "mws":
        algorithm_params = {
            **algorithm_params,
            "num_particles": args.num_particles,
            "memory_size": args.memory_size,
        }
    else:
        raise NotImplementedError
    train.train(
        args.algorithm,
        generative_model,
        inference_network,
        memory,
        data_loader,
        test_data_loader,
        optimizer,
        algorithm_params,
        stats,
        run_args=args,
    )


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test-run", action="store_true", help="")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--condition-on-alphabet", action="store_true", help=" ")

    # data
    parser.add_argument("--dataset", default="omniglot", help=" ")
    parser.add_argument("--data-location", default="local", help=" ")
    parser.add_argument("--batch-size", type=int, default=250, help=" ")
    parser.add_argument("--small-dataset", action="store_true", help="use a small dataset")
    parser.add_argument("--dataset-size", default=None, type=int)

    # init
    # parser.add_argument('--uniform-mixture', action='store_true', help=' ')
    parser.add_argument(
        "--likelihood",
        default="learned-affine",
        choices=["bernoulli", "learned-affine", "affine", "classify"],
        help=" ",
    )
    parser.add_argument("--num-arcs", type=int, default=10, help=" ")

    parser.add_argument("--p-lstm-hidden-size", type=int, default=4, help=" ")
    parser.add_argument("--p-uniform-mixture", default=0.0, type=float)

    parser.add_argument("--q-lstm-hidden-size", type=int, default=64, help=" ")
    parser.add_argument("--q-uniform-mixture", default=0.0, type=float)

    parser.add_argument("--num-particles", type=int, default=10, help=" ")
    parser.add_argument("--memory-size", type=int, default=10, help=" ")
    parser.add_argument("--obs-embedding-dim", type=int, default=64, help=" ")
    parser.add_argument("--num-primitives", type=int, default=64, help=" ")
    parser.add_argument("--initial-max-curve", type=float, default=0.3, help=" ")
    parser.add_argument("--big-arcs", action="store_true", help=" ")

    # train
    parser.add_argument("--algorithm", default="mws", choices=["mws", "rws", "vimco"], help=" ")
    parser.add_argument("--prior-lr-factor", default=1.0, type=float)
    parser.add_argument("--pretrain-iterations", type=int, default=10000, help=" ")
    parser.add_argument("--prior-anneal-iterations", type=int, default=0, help=" ")
    parser.add_argument("--num-iterations", type=int, default=200000, help=" ")
    parser.add_argument("--log-interval", type=int, default=1, help=" ")
    parser.add_argument("--save-interval", type=int, default=1000, help=" ")
    parser.add_argument("--test-interval", type=int, default=999999, help=" ")
    parser.add_argument("--test-batch-size", type=int, default=5, help=" ")
    parser.add_argument("--legacy-index", action="store_true")
    parser.add_argument("--test-num-particles", type=int, default=1000, help=" ")
    parser.add_argument("--checkpoint-path-prefix", default="checkpoint", help=" ")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.memory_size is None:
        args.memory_size = args.num_particles

    while True:
        try:
            main(args)
            break
        except RuntimeError as e:
            if "CUDA out of memory" in repr(e):
                args.batch_size = args.batch_size - 1
                print(e)
                print("Decreasing batch_size to: {}".format(args.batch_size))
                if args.batch_size == 0:
                    raise RuntimeError("batch size got decreased to 0")
                else:
                    continue
            else:
                raise e
