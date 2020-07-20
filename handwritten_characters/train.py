import torch
import losses
import util
import math
import shutil


def train(
    algorithm,
    generative_model,
    inference_network,
    memory,
    data_loader,
    test_data_loader,
    optimizer,
    algorithm_params,
    stats,
    run_args=None,
):
    checkpoint_iterations = util.get_checkpoint_iterations(algorithm_params["num_iterations"])
    device = next(generative_model.parameters()).device

    if memory is not None:
        util.logging.info(
            f"Size of MWS's memory of shape {memory.shape}: "
            f"{memory.element_size() * memory.nelement() / 1e6:.2f} MB"
        )
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    iteration = len(stats.theta_losses)

    while iteration < algorithm_params["num_iterations"]:
        for obs_id, obs in data_loader:
            iteration += 1

            # Loss
            if algorithm == "rws" or algorithm == "vimco":
                if algorithm == "rws":
                    loss_fn = losses.get_rws_loss
                elif algorithm == "vimco":
                    loss_fn = losses.get_vimco_loss
                loss, theta_loss, phi_loss = loss_fn(
                    generative_model,
                    inference_network,
                    obs,
                    obs_id,
                    algorithm_params["num_particles"],
                )
            elif algorithm == "mws":
                (
                    loss,
                    theta_loss,
                    phi_loss,
                    prior_loss,
                    accuracy,
                    novel_proportion,
                    new_map,
                ) = losses.get_mws_loss(
                    generative_model,
                    inference_network,
                    memory,
                    obs,
                    obs_id,
                    algorithm_params["num_particles"],
                )

            # Backprop
            loss.backward()

            # Step
            optimizer.step()
            optimizer.zero_grad()

            stats.theta_losses.append(theta_loss)
            stats.phi_losses.append(phi_loss)

            if algorithm == "mws":
                stats.prior_losses.append(prior_loss)
                if accuracy is not None:
                    stats.accuracies.append(accuracy)
                if novel_proportion is not None:
                    stats.novel_proportions.append(novel_proportion)
                if new_map is not None:
                    stats.new_maps.append(new_map)

            if iteration % algorithm_params["log_interval"] == 0:
                if algorithm == "mws":
                    util.logging.info(
                        "it. {}/{} | prior loss = {:.2f} | theta loss = {:.2f} | "
                        "phi loss = {:.2f} | accuracy = {}% | novel = {}% | new map = {}% "
                        "| last log_p = {} | last kl = {} | GPU memory = {:.2f} MB".format(
                            iteration,
                            algorithm_params["num_iterations"],
                            prior_loss,
                            theta_loss,
                            phi_loss,
                            accuracy * 100 if accuracy is not None else None,
                            novel_proportion * 100 if novel_proportion is not None else None,
                            new_map * 100 if new_map is not None else None,
                            "N/A" if len(stats.log_ps) == 0 else stats.log_ps[-1],
                            "N/A" if len(stats.kls) == 0 else stats.kls[-1],
                            (
                                torch.cuda.max_memory_allocated(device=device) / 1e6
                                if device.type == "cuda"
                                else 0
                            ),
                        )
                    )
                elif algorithm == "rws" or algorithm == "vimco":
                    util.logging.info(
                        "it. {}/{} | theta loss = {:.2f} | "
                        "phi loss = {:.2f} | last log_p = {} | "
                        "last kl = {} | GPU memory = {:.2f} MB".format(
                            iteration,
                            algorithm_params["num_iterations"],
                            theta_loss,
                            phi_loss,
                            "N/A" if len(stats.log_ps) == 0 else stats.log_ps[-1],
                            "N/A" if len(stats.kls) == 0 else stats.kls[-1],
                            (
                                torch.cuda.max_memory_allocated(device=device) / 1e6
                                if device.type == "cuda"
                                else 0
                            ),
                        )
                    )
            if iteration % algorithm_params["test_interval"] == 0:
                log_p, kl = eval_gen_inf(
                    generative_model,
                    inference_network,
                    test_data_loader,
                    algorithm_params["test_num_particles"],
                )
                stats.log_ps.append(log_p)
                stats.kls.append(kl)
            if (
                iteration % algorithm_params["save_interval"] == 0
                or iteration in checkpoint_iterations
            ):
                util.save_checkpoint(
                    algorithm_params["checkpoint_path"],
                    generative_model,
                    inference_network,
                    optimizer,
                    memory,
                    stats,
                    run_args=run_args,
                )
                if iteration in checkpoint_iterations:
                    checkpoint_path_current = util.get_checkpoint_path(
                        run_args, checkpoint_iteration=iteration,
                    )
                    shutil.copy(algorithm_params["checkpoint_path"], checkpoint_path_current)
            if iteration == algorithm_params["num_iterations"]:
                break

    util.save_checkpoint(
        algorithm_params["checkpoint_path"],
        generative_model,
        inference_network,
        optimizer,
        memory,
        stats,
        run_args=run_args,
    )


def get_log_p_and_kl(generative_model, inference_network, obs, obs_id, num_particles):
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
    latent_dist = inference_network.get_latent_dist(obs)
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


def eval_gen_inf(generative_model, inference_network, data_loader, num_particles):
    log_p_total = 0
    kl_total = 0
    reconstruction_log_prob_total = 0
    num_obs = len(data_loader.dataset)
    for obs_id, obs in data_loader:
        log_p, kl, reconstruction_log_prob = get_log_p_and_kl(
            generative_model, inference_network, obs, obs_id, num_particles
        )
        log_p_total += torch.sum(log_p).item() / num_obs
        kl_total += torch.sum(kl).item() / num_obs
        reconstruction_log_prob_total += torch.sum(reconstruction_log_prob).item() / num_obs
    return log_p_total, reconstruction_log_prob_total


def train_sleep(generative_model, inference_network, num_samples, num_iterations, log_interval):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    sleep_losses = []
    device = next(generative_model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    util.logging.info("Pretraining with sleep")
    iteration = 0
    while iteration < num_iterations:
        optimizer_phi.zero_grad()
        sleep_phi_loss = losses.get_sleep_loss(generative_model, inference_network, num_samples)
        sleep_phi_loss.backward()
        optimizer_phi.step()

        sleep_losses.append(sleep_phi_loss.item())
        iteration += 1
        # by this time, we have gone through `iteration` iterations
        if iteration % log_interval == 0:
            util.logging.info(
                "it. {}/{} | sleep loss = {:.2f} | "
                "GPU memory = {:.2f} MB".format(
                    iteration,
                    num_iterations,
                    sleep_losses[-1],
                    (
                        torch.cuda.max_memory_allocated(device=device) / 1e6
                        if device.type == "cuda"
                        else 0
                    ),
                )
            )
        if iteration == num_iterations:
            break
