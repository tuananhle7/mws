import torch

import util
import train
from pathlib import Path


def run(args):
    # set up args
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        args.cuda = True
    else:
        device = torch.device("cpu")
        args.cuda = False

    util.print_with_time("args = {}".format(args))

    # init
    util.print_with_time("init")
    true_cluster_cov = torch.eye(args.num_dim, device=device) * 0.03
    generative_model, inference_network, true_generative_model = util.init(
        args.num_data, args.num_dim, true_cluster_cov, device
    )
    util.set_seed(args.seed)

    # data
    util.print_with_time("data")
    train_data, test_data = util.load_data()
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True
    )

    # train
    util.print_with_time("train")
    checkpoint_path = "checkpoints/{}_{}_{}_{}.pt".format(
        args.checkpoint_path_prefix, args.algorithm, args.num_particles, args.seed
    )
    Path("checkpoints/").mkdir(parents=True, exist_ok=True)
    if args.algorithm == "mws":
        (
            theta_losses,
            phi_losses,
            cluster_cov_distances,
            test_log_ps,
            test_log_ps_true,
            test_kl_qps,
            test_kl_pqs,
            test_kl_qps_true,
            test_kl_pqs_true,
            train_log_ps,
            train_log_ps_true,
            train_kl_qps,
            train_kl_pqs,
            train_kl_qps_true,
            train_kl_pqs_true,
            train_kl_memory_ps,
            train_kl_memory_ps_true,
            memory,
            reweighted_train_kl_qps,
            reweighted_train_kl_qps_true,
        ) = train.train_mws(
            generative_model,
            inference_network,
            data_loader,
            args.num_iterations,
            args.memory_size,
            true_cluster_cov,
            test_data_loader,
            args.test_num_particles,
            true_generative_model,
            checkpoint_path,
        )
    elif args.algorithm == "rmws":
        (
            theta_losses,
            phi_losses,
            cluster_cov_distances,
            test_log_ps,
            test_log_ps_true,
            test_kl_qps,
            test_kl_pqs,
            test_kl_qps_true,
            test_kl_pqs_true,
            train_log_ps,
            train_log_ps_true,
            train_kl_qps,
            train_kl_pqs,
            train_kl_qps_true,
            train_kl_pqs_true,
            train_kl_memory_ps,
            train_kl_memory_ps_true,
            memory,
            reweighted_train_kl_qps,
            reweighted_train_kl_qps_true,
        ) = train.train_mws(
            generative_model,
            inference_network,
            data_loader,
            args.num_iterations,
            args.memory_size,
            true_cluster_cov,
            test_data_loader,
            args.test_num_particles,
            true_generative_model,
            checkpoint_path,
            reweighted=True,
        )
    elif args.algorithm == "rws":
        (
            theta_losses,
            phi_losses,
            cluster_cov_distances,
            test_log_ps,
            test_log_ps_true,
            test_kl_qps,
            test_kl_pqs,
            test_kl_qps_true,
            test_kl_pqs_true,
            train_log_ps,
            train_log_ps_true,
            train_kl_qps,
            train_kl_pqs,
            train_kl_qps_true,
            train_kl_pqs_true,
            train_kl_memory_ps,
            train_kl_memory_ps_true,
            memory,
            reweighted_train_kl_qps,
            reweighted_train_kl_qps_true,
        ) = train.train_rws(
            generative_model,
            inference_network,
            data_loader,
            args.num_iterations,
            args.num_particles,
            true_cluster_cov,
            test_data_loader,
            args.test_num_particles,
            true_generative_model,
            checkpoint_path,
        )
    elif args.algorithm == "vimco":
        (
            theta_losses,
            phi_losses,
            cluster_cov_distances,
            test_log_ps,
            test_log_ps_true,
            test_kl_qps,
            test_kl_pqs,
            test_kl_qps_true,
            test_kl_pqs_true,
            train_log_ps,
            train_log_ps_true,
            train_kl_qps,
            train_kl_pqs,
            train_kl_qps_true,
            train_kl_pqs_true,
            train_kl_memory_ps,
            train_kl_memory_ps_true,
            memory,
            reweighted_train_kl_qps,
            reweighted_train_kl_qps_true,
        ) = train.train_vimco(
            generative_model,
            inference_network,
            data_loader,
            args.num_iterations,
            args.num_particles,
            true_cluster_cov,
            test_data_loader,
            args.test_num_particles,
            true_generative_model,
            checkpoint_path,
        )

    # save model
    util.save_checkpoint(
        checkpoint_path,
        generative_model,
        inference_network,
        theta_losses,
        phi_losses,
        cluster_cov_distances,
        test_log_ps,
        test_log_ps_true,
        test_kl_qps,
        test_kl_pqs,
        test_kl_qps_true,
        test_kl_pqs_true,
        train_log_ps,
        train_log_ps_true,
        train_kl_qps,
        train_kl_pqs,
        train_kl_qps_true,
        train_kl_pqs_true,
        train_kl_memory_ps,
        train_kl_memory_ps_true,
        memory,
        reweighted_train_kl_qps,
        reweighted_train_kl_qps_true,
    )


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--algorithm", default="mws", help="rws, mws, vimco, rmws")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--batch-size", type=int, default=20, help=" ")
    parser.add_argument("--num-dim", type=int, default=2, help=" ")
    parser.add_argument("--num-data", type=int, default=7, help=" ")
    # parser.add_argument('--num-train', type=int, default=100, help=' ')
    parser.add_argument("--num-iterations", type=int, default=50000, help=" ")
    parser.add_argument("--num-particles", type=int, default=5, help=" ")
    parser.add_argument("--memory-size", type=int, default=5, help=" ")
    # parser.add_argument('--num-test', type=int, default=100, help=' ')
    parser.add_argument("--test-num-particles", type=int, default=5000, help=" ")
    parser.add_argument("--checkpoint-path-prefix", default="checkpoint", help=" ")
    parser.add_argument("--seed", type=int, default=1, help=" ")
    return parser


if __name__ == "__main__":
    run(get_args_parser().parse_args())
