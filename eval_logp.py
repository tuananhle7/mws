import torch
import util
import test
import data
from pathlib import Path


def main(args):
    util.logging.info("Running eval_logp")
    util.print_args(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        util.logging.info("Using CUDA")
    else:
        device = torch.device("cpu")
        util.logging.info("Using CPU")

    # Load data
    util.logging.info("loading data")
    data_location = "om"
    (
        data_train,
        data_valid,
        data_test,
        target_train,
        target_valid,
        target_test,
    ) = data.load_binarized_omniglot_with_targets(location=data_location)
    if args.mode == "train":
        data_ = data_train
    elif args.mode == "test":
        data_ = data_test
    data_ = data_[: args.num_data]
    data_loader_ = data.get_data_loader(data_, args.batch_size, device, ids=True)
    test_data_loader_ = data.get_data_loader(data_, args.test_batch_size, device, ids=True)

    (
        generative_model,
        inference_network,
        optimizer,
        memory,
        stats,
        run_args,
    ) = util.load_checkpoint(args.checkpoint_path, device)

    if args.algorithm == "vimco":
        log_p, losses = test.evaluate_logp_vimco(
            generative_model,
            data_loader_,
            test_data_loader_,
            args.lstm_hidden_size,
            args.num_particles,
            args.test_num_particles,
            args.num_iterations,
            args.log_interval,
        )
    elif args.algorithm == "rws":
        log_p, losses = test.evaluate_logp_rws(
            generative_model,
            data_loader_,
            test_data_loader_,
            args.lstm_hidden_size,
            args.num_particles,
            args.test_num_particles,
            args.num_iterations,
            args.log_interval,
        )
    elif args.algorithm == "sleep":
        obs_embedding_dim = 64
        num_samples_ = args.num_particles * args.batch_size
        log_p, losses = test.evaluate_logp_sleep(
            generative_model,
            data_loader_,
            test_data_loader_,
            args.lstm_hidden_size,
            obs_embedding_dim,
            num_samples_,
            args.test_num_particles,
            args.num_iterations,
            args.log_interval,
        )

    util.logging.info(f"logp of {args.checkpoint_path} based on {args.algorithm} = {log_p}")
    save_path = util.get_logp_path(run_args, args.mode, args.algorithm)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save([log_p, losses], save_path)
    util.logging.info(f"saved log_ps to {save_path}")


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-path", default="", help=" ")
    parser.add_argument("--mode", default="test", help="test or train")
    parser.add_argument("--algorithm", default="sleep", help="sleep, rws or vimco")
    parser.add_argument("--num-data", type=int, default=50, help=" ")

    # parameters used to train the inference network
    # while fixing the generative model
    parser.add_argument("--num-particles", type=int, default=100, help=" ")
    parser.add_argument("--num-iterations", type=int, default=20000, help=" ")
    parser.add_argument("--log-interval", type=int, default=1, help=" ")
    parser.add_argument("--batch-size", type=int, default=50, help=" ")
    parser.add_argument("--lstm-hidden-size", type=int, default=128, help=" ")

    # params used to evaluate the generative model's
    # log p \approx IWAE objective
    # after the inference network has finished training
    parser.add_argument("--test-batch-size", type=int, default=1, help=" ")
    parser.add_argument("--test-num-particles", type=int, default=5000, help=" ")
    return parser


if __name__ == "__main__":
    main(get_arg_parser().parse_args())
