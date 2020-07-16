import os
import models
import torch
import math
import logging
from pathlib import Path
import collections
import pprint
import subprocess
import getpass
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def get_path_base_from_args(args):
    if args.condition_on_alphabet:
        prefix = "a_"
    else:
        prefix = ""

    if args.small_dataset:
        suffix = "small"
    else:
        suffix = "large"

    return f"{prefix}{args.algorithm}_{args.num_particles}_{suffix}"


def get_save_job_name_from_args(args):
    return get_path_base_from_args(args)


def get_save_dir_from_path_base(path_base):
    return f"save/{path_base}"


def get_save_dir(args):
    return get_save_dir_from_path_base(get_path_base_from_args(args))


def get_checkpoint_path(args, checkpoint_iteration=-1):
    return get_checkpoint_path_from_path_base(get_path_base_from_args(args), checkpoint_iteration)


def get_checkpoint_path_from_path_base(path_base, checkpoint_iteration=-1):
    checkpoints_dir = f"{get_save_dir_from_path_base(path_base)}/checkpoints"
    if checkpoint_iteration == -1:
        return f"{checkpoints_dir}/latest.pt"
    else:
        return f"{checkpoints_dir}/{checkpoint_iteration}.pt"


def get_logp_path(args, test_mode, test_algorithm):
    return f"{get_save_dir(args)}/logp/{test_mode}_{test_algorithm}.pt"


def logit_uniform_mixture(values, delta, dim=-1):
    """
    Given logits for a categorical distribution D, return the logits
    for a mixture distribution (1-delta)*D + delta*uniform
    """
    n = values.shape[dim]
    term1 = values + math.log(1 - delta)
    term2 = torch.logsumexp(values + math.log(delta / n), dim=dim, keepdim=True).expand_as(term1)
    logits = torch.stack([term1, term2]).logsumexp(dim=0)
    return logits


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def init_memory(num_data, memory_size, num_arcs, num_primitives, device):
    logging.info("Initializing MWS's memory")
    memory = []
    for _ in range(num_data):
        while True:
            x = torch.stack(
                [
                    torch.randint(num_primitives, (memory_size, num_arcs), device=device),
                    torch.randint(2, (memory_size, num_arcs), device=device),
                ],
                dim=-1,
            )
            if len(torch.unique(x, dim=0)) == memory_size:
                memory.append(x)
                break
    return torch.stack(memory)


def init_optimizer(generative_model, inference_network, prior_lr_factor):
    lr = 1e-3
    optimizer = torch.optim.Adam(
        [
            {"params": generative_model._prior.parameters(), "lr": lr * prior_lr_factor},
            {"params": generative_model._likelihood.parameters(), "lr": lr},
            {"params": inference_network.parameters(), "lr": lr},
        ]
    )
    return optimizer


def init(run_args, device):
    generative_model = models.GenerativeModel(
        run_args.num_primitives,
        run_args.initial_max_curve,
        run_args.big_arcs,
        run_args.p_lstm_hidden_size,
        run_args.num_rows,
        run_args.num_cols,
        run_args.num_arcs,
        run_args.likelihood,
        run_args.p_uniform_mixture,
        use_alphabet=run_args.condition_on_alphabet,
    ).to(device)
    inference_network = models.InferenceNetwork(
        run_args.num_primitives,
        run_args.q_lstm_hidden_size,
        run_args.num_rows,
        run_args.num_cols,
        run_args.num_arcs,
        run_args.obs_embedding_dim,
        run_args.q_uniform_mixture,
        use_alphabet=run_args.condition_on_alphabet,
    ).to(device)
    optimizer = init_optimizer(generative_model, inference_network, 1,)

    stats = Stats([], [], [], [], [], [], [], [])

    if "mws" in run_args.algorithm:
        memory = init_memory(
            run_args.num_train_data,
            run_args.memory_size,
            generative_model.num_arcs,
            generative_model.num_primitives,
            device,
        )
    else:
        memory = None

    return generative_model, inference_network, optimizer, memory, stats


def save_checkpoint(
    path,
    # models
    generative_model,
    inference_network,
    optimizer,
    memory,
    # stats
    stats,
    # run args
    run_args=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # models
            "generative_model_state_dict": generative_model.state_dict(),
            "inference_network_state_dict": inference_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "memory": memory,
            # stats
            "stats": stats,
            # run args
            "run_args": run_args,
        },
        path,
    )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    generative_model, inference_network, optimizer, _, _ = init(checkpoint["run_args"], device)

    generative_model.load_state_dict(checkpoint["generative_model_state_dict"])
    inference_network.load_state_dict(checkpoint["inference_network_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    memory = checkpoint["memory"]
    stats = checkpoint["stats"]
    run_args = checkpoint["run_args"]
    return generative_model, inference_network, optimizer, memory, stats, run_args


Stats = collections.namedtuple(
    "Stats",
    [
        "log_ps",
        "kls",
        "theta_losses",
        "phi_losses",
        "accuracies",
        "novel_proportions",
        "new_maps",
        "prior_losses",
    ],
)


def get_checkpoint_iterations(num_iterations):
    # Five checkpoints for each power of 10 iterations
    f = 10 ** (1 / 5) + 1e-10
    return set(
        [int(f ** n) for n in range(int(math.log(num_iterations) / math.log(f)) + 1)]
        + list(range(0, num_iterations, 5000))
    )


def print_args(args):
    logging.info("args:")
    pp = pprint.PrettyPrinter(indent=2, width=80, compact=True)
    pp.pprint(vars(args))


def cancel_all_my_non_bash_jobs():
    logging.info("Cancelling all non-bash jobs.")
    jobs_status = (
        subprocess.check_output(f"squeue -u {getpass.getuser()}", shell=True)
        .decode()
        .split("\n")[1:-1]
    )
    non_bash_job_ids = []
    for job_status in jobs_status:
        if not ("bash" in job_status.split() or "zsh" in job_status.split()):
            non_bash_job_ids.append(job_status.split()[0])
    if len(non_bash_job_ids) > 0:
        cmd = "scancel {}".format(" ".join(non_bash_job_ids))
        logging.info(cmd)
        logging.info(subprocess.check_output(cmd, shell=True).decode())
    else:
        logging.info("No non-bash jobs to cancel.")


def get_checkpoint_paths(checkpoint_iteration=-1):
    save_dir = "./save/"
    for path_base in sorted(os.listdir(save_dir)):
        yield get_checkpoint_path_from_path_base(path_base, checkpoint_iteration)
