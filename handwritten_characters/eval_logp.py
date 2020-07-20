import torch
import util
import data
from pathlib import Path
import math
import torch.nn as nn
import torch.nn.functional as F
import models
import losses
import sweep
import numpy as np


class InferenceNetworkTest(nn.Module):
    def __init__(
        self, num_primitives, lstm_hidden_size, num_rows, num_cols, num_arcs, num_test_data
    ):
        super(InferenceNetworkTest, self).__init__()
        self.num_primitives = num_primitives
        self.obs_embedding_dim = num_test_data
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_input_size = num_primitives + 2 + self.obs_embedding_dim
        self.lstm_cell = nn.LSTMCell(
            input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size
        )
        self.linear = nn.Linear(self.lstm_hidden_size, num_primitives + 2)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_arcs = num_arcs
        self.register_buffer("start_point", torch.tensor([0.5, 0.5]))

    def get_obs_embedding(self, obs_id):
        """
        Args:
            obs: tensor [batch_size]

        Returns: tensor [batch_size, obs_embedding_dim]
        """
        return F.one_hot(obs_id, num_classes=self.obs_embedding_dim).float()

    def get_latent_dist(self, obs_id):
        """Args:
            obs: tensor of shape [batch_size]

        Returns: distribution with batch_shape [batch_size] and
            event_shape [num_arcs, 2]
        """
        return models.InferenceNetworkIdsAndOnOffsDistribution(
            self.get_obs_embedding(obs_id), self.lstm_cell, self.linear, self.num_arcs
        )

    def sample_from_latent_dist(self, latent_dist, num_particles):
        """Samples from q(latent | obs)

        Args:
            latent_dist: distribution with batch_shape [batch_size] and
                event_shape [num_arcs, 2]
            num_particles: int

        Returns:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        """
        return latent_dist.sample((num_particles,))

    def sample(self, obs_id, num_particles):
        """Samples from q(latent | obs)

        Args:
            obs: tensor of shape [batch_size]
            num_particles: int

        Returns:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        """
        latent_dist = self.get_latent_dist(obs_id)
        return self.sample_from_latent_dist(latent_dist, num_particles)

    def get_log_prob_from_latent_dist(self, latent_dist, latent):
        """Log q(latent | obs).

        Args:
            latent_dist: distribution with batch_shape [batch_size] and
                event_shape [num_arcs, 2]
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return latent_dist.log_prob(latent)

    def get_log_prob(self, latent, obs_id):
        """Log q(latent | obs).

        Args:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
            obs: tensor of shape [batch_size]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return self.get_log_prob_from_latent_dist(self.get_latent_dist(obs_id), latent)


def get_log_p_and_kl_(generative_model, inference_network, obs, obs_id, num_particles):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_data * num_dim]
        num_particles: int

    Returns:
        log_p: tensor [batch_size]
        kl: tensor [batch_size]
    """
    latent_dist = inference_network.get_latent_dist(obs_id)
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles)
    latent_log_prob, obs_log_prob = generative_model.get_log_probss(latent, obs, obs_id)
    latent_log_prob = latent_log_prob.transpose(0, 1)
    obs_log_prob = obs_log_prob.transpose(0, 1)
    log_joint = latent_log_prob + obs_log_prob
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_joint - log_q

    log_p = torch.logsumexp(log_weight, dim=1) - math.log(num_particles)
    elbo = torch.mean(log_weight, dim=1)
    kl = log_p - elbo
    reconstruction_log_prob = torch.mean(obs_log_prob, dim=1)

    return log_p, kl, reconstruction_log_prob


def eval_gen_inf_(generative_model, inference_network, data_loader, num_particles):
    log_p_total = 0
    kl_total = 0
    reconstruction_log_prob_total = 0
    num_obs = len(data_loader.dataset)
    for obs_id, obs in data_loader:
        log_p, kl, reconstruction_log_prob = get_log_p_and_kl_(
            generative_model, inference_network, obs, obs_id, num_particles
        )
        log_p_total += torch.sum(log_p).item() / num_obs
        kl_total += torch.sum(kl).item() / num_obs
        reconstruction_log_prob_total += torch.sum(reconstruction_log_prob).item() / num_obs
    return log_p_total, reconstruction_log_prob_total


def eval_gen_inf_sleep(generative_model, inference_network, data_loader, num_particles):
    log_p_total = 0
    kl_total = 0
    reconstruction_log_prob_total = 0
    num_obs = len(data_loader.dataset)
    for obs_id, obs in data_loader:
        log_p, kl, reconstruction_log_prob = get_log_p_and_kl_(
            generative_model, inference_network, obs, obs, num_particles
        )
        log_p_total += torch.sum(log_p).item() / num_obs
        kl_total += torch.sum(kl).item() / num_obs
        reconstruction_log_prob_total += torch.sum(reconstruction_log_prob).item() / num_obs
    return log_p_total, reconstruction_log_prob_total


def get_vimco_loss_(generative_model, inference_network, obs, obs_id, num_particles):
    """Almost twice faster version of VIMCO loss (measured for batch_size = 24,
        num_particles = 1000). Inspired by Adam Kosiorek's implementation.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_rows, num_cols]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    latent_dist = inference_network.get_latent_dist(obs_id)
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles)
    log_p = generative_model.get_log_prob(latent, obs, obs_id).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q

    # shape [batch_size, num_particles]
    # log_weight_[b, k] = 1 / (K - 1) \sum_{\ell \neq k} \log w_{b, \ell}
    log_weight_ = (torch.sum(log_weight, dim=1, keepdim=True) - log_weight) / (num_particles - 1)

    # shape [batch_size, num_particles, num_particles]
    # temp[b, k, k_] =
    #     log_weight_[b, k]     if k == k_
    #     log_weight[b, k]      otherwise
    temp = log_weight.unsqueeze(-1) + torch.diag_embed(log_weight_ - log_weight)

    # this is the \Upsilon_{-k} term below equation 3
    # shape [batch_size, num_particles]
    control_variate = torch.logsumexp(temp, dim=1) - math.log(num_particles)

    log_evidence = torch.logsumexp(log_weight, dim=1) - math.log(num_particles)
    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(
        torch.sum((log_evidence.unsqueeze(-1) - control_variate).detach() * log_q, dim=1)
    )

    return loss, elbo


def train_vimco_(
    generative_model,
    inference_network,
    data_loader,
    test_data_loader,
    num_particles,
    test_num_particles,
    num_iterations,
    log_interval,
):
    optimizer = torch.optim.Adam(inference_network.parameters())
    theta_losses, phi_losses, log_ps, kls = [], [], [], []
    device = next(generative_model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    iteration = 0
    while iteration < num_iterations:
        for obs_id, obs in data_loader:
            # loss
            optimizer.zero_grad()
            loss, elbo = get_vimco_loss_(
                generative_model, inference_network, obs, obs_id, num_particles
            )
            loss.backward(retain_graph=True)
            optimizer.step()

            theta_losses.append(-elbo.item())
            phi_losses.append(loss.item())
            iteration += 1
            # by this time, we have gone through `iteration` iterations
            if iteration % log_interval == 0:
                util.logging.info(
                    "it. {}/{} | theta loss = {:.2f} | "
                    "phi loss = {:.2f} | last log_p = {} | "
                    "last kl = {} | GPU memory = {:.2f} MB".format(
                        iteration,
                        num_iterations,
                        theta_losses[-1],
                        phi_losses[-1],
                        "N/A" if len(log_ps) == 0 else log_ps[-1],
                        "N/A" if len(kls) == 0 else kls[-1],
                        (
                            torch.cuda.max_memory_allocated(device=device) / 1e6
                            if device.type == "cuda"
                            else 0
                        ),
                    )
                )
            if iteration == num_iterations:
                break

    log_p, kl = eval_gen_inf_(
        generative_model, inference_network, test_data_loader, test_num_particles
    )

    return log_p, theta_losses


def train_rws_(
    generative_model,
    inference_network,
    data_loader,
    test_data_loader,
    num_particles,
    test_num_particles,
    num_iterations,
    log_interval,
):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    theta_losses, phi_losses, log_ps, kls = [], [], [], []
    device = next(generative_model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    iteration = 0
    while iteration < num_iterations:
        for obs_id, obs in data_loader:
            latent_dist = inference_network.get_latent_dist(obs_id)
            latent = inference_network.sample_from_latent_dist(latent_dist, num_particles)
            log_p = generative_model.get_log_prob(latent, obs, obs_id).transpose(0, 1)
            log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, latent).transpose(
                0, 1
            )
            log_weight = log_p - log_q

            # wake phi
            optimizer_phi.zero_grad()
            wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)
            wake_phi_loss.backward()
            optimizer_phi.step()

            phi_losses.append(wake_phi_loss.item())
            iteration += 1
            # by this time, we have gone through `iteration` iterations
            if iteration % log_interval == 0:
                util.logging.info(
                    "it. {}/{} | "
                    "phi loss = {:.2f} | last log_p = {} | "
                    "last kl = {} | GPU memory = {:.2f} MB".format(
                        iteration,
                        num_iterations,
                        phi_losses[-1],
                        "N/A" if len(log_ps) == 0 else log_ps[-1],
                        "N/A" if len(kls) == 0 else kls[-1],
                        (
                            torch.cuda.max_memory_allocated(device=device) / 1e6
                            if device.type == "cuda"
                            else 0
                        ),
                    )
                )
            if iteration == num_iterations:
                break

    log_p, kl = eval_gen_inf_(
        generative_model, inference_network, test_data_loader, test_num_particles
    )
    return log_p, phi_losses


def get_sleep_loss_(generative_model, inference_network, num_samples=1):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """

    # latent, obs = generative_model.sample_latent_and_obs(
    #     num_samples=num_samples)
    device = next(generative_model.parameters()).device
    if generative_model.use_alphabet:
        alphabet = (
            torch.distributions.OneHotCategorical(logits=torch.ones(50, device=device).float())
            .sample((num_samples,))
            .contiguous()
        )
        latent, obs = generative_model.sample_latent_and_obs(alphabet=alphabet, num_samples=1)
        latent = latent[0]
        obs = obs[0]
    else:
        alphabet = None
        latent, obs = generative_model.sample_latent_and_obs(
            alphabet=alphabet, num_samples=num_samples
        )
    if generative_model.use_alphabet:
        obs = (obs, alphabet)

    return -torch.mean(inference_network.get_log_prob(latent, obs))


def train_sleep_(
    generative_model,
    inference_network,
    data_loader,
    test_data_loader,
    num_samples,
    test_num_particles,
    num_iterations,
    log_interval,
):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    losses, log_ps, kls = [], [], []
    device = next(generative_model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    iteration = 0
    while iteration < num_iterations:
        optimizer_phi.zero_grad()
        sleep_phi_loss = get_sleep_loss_(generative_model, inference_network, num_samples)
        sleep_phi_loss.backward()
        optimizer_phi.step()

        losses.append(sleep_phi_loss.item())
        iteration += 1
        # by this time, we have gone through `iteration` iterations
        if iteration % log_interval == 0:
            util.logging.info(
                "it. {}/{} | theta loss = {:.2f} | "
                "phi loss = {:.2f} | last log_p = {} | "
                "last kl = {} | GPU memory = {:.2f} MB".format(
                    iteration,
                    num_iterations,
                    losses[-1],
                    losses[-1],
                    "N/A" if len(log_ps) == 0 else log_ps[-1],
                    "N/A" if len(kls) == 0 else kls[-1],
                    (
                        torch.cuda.max_memory_allocated(device=device) / 1e6
                        if device.type == "cuda"
                        else 0
                    ),
                )
            )
        if iteration == num_iterations:
            break

    log_p, kl = eval_gen_inf_sleep(
        generative_model, inference_network, test_data_loader, test_num_particles
    )

    return log_p, losses


def evaluate_logp_vimco(
    generative_model,
    data_loader,
    test_data_loader,
    lstm_hidden_size,
    num_particles,
    test_num_particles,
    num_iterations,
    log_interval,
):
    device = next(generative_model.parameters()).device
    inference_network = InferenceNetworkTest(
        generative_model.num_primitives,
        lstm_hidden_size,
        generative_model.num_rows,
        generative_model.num_cols,
        generative_model.num_arcs,
        len(data_loader.dataset),
    ).to(device)

    logp, losses = train_vimco_(
        generative_model,
        inference_network,
        data_loader=data_loader,
        test_data_loader=test_data_loader,
        num_particles=num_particles,
        test_num_particles=test_num_particles,
        num_iterations=num_iterations,
        log_interval=log_interval,
    )

    return logp, losses


def evaluate_logp_rws(
    generative_model,
    data_loader,
    test_data_loader,
    lstm_hidden_size,
    num_particles,
    test_num_particles,
    num_iterations,
    log_interval,
):
    device = next(generative_model.parameters()).device
    inference_network = InferenceNetworkTest(
        generative_model.num_primitives,
        lstm_hidden_size,
        generative_model.num_rows,
        generative_model.num_cols,
        generative_model.num_arcs,
        len(data_loader.dataset),
    ).to(device)

    logp, losses = train_rws_(
        generative_model,
        inference_network,
        data_loader=data_loader,
        test_data_loader=test_data_loader,
        num_particles=num_particles,
        test_num_particles=test_num_particles,
        num_iterations=num_iterations,
        log_interval=log_interval,
    )

    return logp, losses


def evaluate_logp_sleep(
    generative_model,
    data_loader,
    test_data_loader,
    lstm_hidden_size,
    obs_embedding_dim,
    num_samples,
    test_num_particles,
    num_iterations,
    log_interval,
):
    device = next(generative_model.parameters()).device
    inference_network = models.InferenceNetwork(
        generative_model.num_primitives,
        lstm_hidden_size,
        generative_model.num_rows,
        generative_model.num_cols,
        generative_model.num_arcs,
        obs_embedding_dim,
        use_alphabet=generative_model.use_alphabet,
    ).to(device)

    logp, losses = train_sleep_(
        generative_model,
        inference_network,
        data_loader,
        test_data_loader,
        num_samples,
        test_num_particles,
        num_iterations,
        log_interval,
    )

    return logp, losses


def report_logp():
    test_algorithms = ["rws", "vimco", "sleep"]
    mode = "test"

    # loading files
    for sweep_args in sweep.get_sweep_argss():
        util.logging.info(util.get_path_base_from_args(sweep_args))
        log_ps = {}
        lossess = {}
        for test_algorithm in test_algorithms:
            save_path = util.get_logp_path(sweep_args, mode, test_algorithm)
            try:
                log_p, losses = torch.load(save_path)
                log_ps[test_algorithm] = log_p
                lossess[test_algorithm] = losses
            except FileNotFoundError:
                util.logging.info("skipping {}".format(save_path))
        if len(log_ps.values()) > 0:
            util.logging.info(f"log_ps = {log_ps}")
            log_ps_values = list(log_ps.values())
            util.logging.info(
                f"best log_p ({test_algorithms[np.argmax(log_ps_values)]}): {np.max(log_ps_values)}"
            )
        print("-------------")


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
        log_p, losses = evaluate_logp_vimco(
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
        log_p, losses = evaluate_logp_rws(
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
        log_p, losses = evaluate_logp_sleep(
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
    parser.add_argument("--report", action="store_true", help="report log p")
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
    args = get_arg_parser().parse_args()
    if args.report:
        report_logp()
    else:
        main(args)
