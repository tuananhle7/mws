import torch
import math
import util


def get_latent_and_log_weight_and_log_q(
    generative_model, inference_network, obs, obs_id, num_particles=1
):
    """Samples latent and computes log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:
        latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """

    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles)
    log_p = generative_model.get_log_prob(latent, obs, obs_id).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q
    return latent, log_weight, log_q


def get_log_weight_and_log_q(generative_model, inference_network, obs, obs_id, num_particles=1):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """

    latent, log_weight, log_q = get_latent_and_log_weight_and_log_q(
        generative_model, inference_network, obs, obs_id, num_particles
    )
    return log_weight, log_q


def get_wake_theta_loss_from_log_weight(log_weight):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    _, num_particles = log_weight.shape
    elbo = torch.mean(torch.logsumexp(log_weight, dim=1) - math.log(num_particles))
    return -elbo, elbo


def get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_rws_loss(generative_model, inference_network, obs, obs_id, num_particles):
    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles)
    log_p = generative_model.get_log_prob(latent, obs, obs_id).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q.detach()

    # wake theta
    wake_theta_loss, elbo = get_wake_theta_loss_from_log_weight(log_weight)

    # wake phi
    wake_phi_loss = get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)

    return wake_theta_loss + wake_phi_loss, wake_theta_loss.item(), wake_phi_loss.item()


def get_vimco_loss(generative_model, inference_network, obs, obs_id, num_particles):
    """Almost twice faster version of VIMCO loss (measured for batch_size = 24,
        num_particles = 1000). Inspired by Adam Kosiorek's implementation.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, obs_id, num_particles
    )

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
    theta_loss = -elbo.item()
    phi_loss = theta_loss

    return loss, theta_loss, phi_loss


def get_mws_loss(generative_model, inference_network, memory, obs, obs_id, num_particles):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        memory: tensor of shape [num_data, memory_size, num_arcs, 2]
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        theta_loss: python float
        phi_loss: python float
        prior_loss, accuracy, novel_proportion, new_map: python float
    """
    memory_size = memory.shape[1]

    # Propose latents from inference network
    latent_dist = inference_network.get_latent_dist(obs)
    # [num_particles, batch_size, num_arcs, 2]
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles).detach()
    batch_size = latent.shape[1]

    # Evaluate log p of proposed latents
    log_prior, log_likelihood = generative_model.get_log_probss(
        latent, obs, obs_id
    )  # [num_particles, batch_size]
    log_p = log_prior + log_likelihood

    # Select latents from memory
    memory_latent = memory[obs_id]  # [batch_size, memory_size, num_arcs, 2]
    memory_latent_transposed = memory_latent.transpose(
        0, 1
    ).contiguous()  # [memory_size, batch_size, num_arcs, 2]

    # Evaluate log p of memory latents
    memory_log_prior, memory_log_likelihood = generative_model.get_log_probss(
        memory_latent_transposed, obs, obs_id
    )  # [memory_size, batch_size]
    memory_log_p = memory_log_prior + memory_log_likelihood
    memory_log_likelihood = None  # don't need this anymore

    # Merge proposed latents and memory latents
    # [memory_size + num_particles, batch_size, num_arcs, 2]
    memory_and_latent = torch.cat([memory_latent_transposed, latent])

    # Evaluate log p of merged latents
    # [memory_size + num_particles, batch_size]
    memory_and_latent_log_p = torch.cat([memory_log_p, log_p])
    memory_and_latent_log_prior = torch.cat([memory_log_prior, log_prior])

    # Compute new map
    new_map = (log_p.max(dim=0).values > memory_log_p.max(dim=0).values).float().mean()

    # Sort log_ps, replace non-unique values with -inf, choose the top k remaining
    # (valid as long as two distinct latents don't have the same logp)
    sorted1, indices1 = memory_and_latent_log_p.sort(dim=0)
    is_same = sorted1[1:] == sorted1[:-1]
    novel_proportion = 1 - (is_same.float().sum() / num_particles / batch_size)
    sorted1[1:].masked_fill_(is_same, float("-inf"))
    sorted2, indices2 = sorted1.sort(dim=0)
    memory_log_p = sorted2[-memory_size:]
    memory_latent
    indices = indices1.gather(0, indices2)[-memory_size:]

    memory_latent = torch.gather(
        memory_and_latent,
        0,
        indices[:, :, None, None]
        .expand(memory_size, batch_size, generative_model.num_arcs, 2)
        .contiguous(),
    )
    memory_log_prior = torch.gather(memory_and_latent_log_prior, 0, indices)

    # Update memory
    # [batch_size, memory_size, num_arcs, 2]
    memory[obs_id] = memory_latent.transpose(0, 1).contiguous()

    # Compute losses
    dist = torch.distributions.Categorical(logits=memory_log_p.t())
    sampled_memory_id = dist.sample()  # [batch_size]
    sampled_memory_id_log_prob = dist.log_prob(sampled_memory_id)  # [batch_size]
    sampled_memory_latent = torch.gather(
        memory_latent,
        0,
        sampled_memory_id[None, :, None, None]
        .expand(1, batch_size, generative_model.num_arcs, 2)
        .contiguous(),
    )  # [1, batch_size, num_arcs, 2]

    log_p = torch.gather(memory_log_p, 0, sampled_memory_id[None])[0]  # [batch_size]
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, sampled_memory_latent)[
        0
    ]  # [batch_size]

    prior_loss = -torch.gather(memory_log_prior, 0, sampled_memory_id[None, :]).mean().detach()
    theta_loss = -torch.mean(log_p - sampled_memory_id_log_prob.detach())  # []
    phi_loss = -torch.mean(log_q)  # []
    loss = theta_loss + phi_loss

    accuracy = None
    return (
        loss,
        theta_loss.item(),
        phi_loss.item(),
        prior_loss.item(),
        accuracy,
        novel_proportion.item(),
        new_map.item(),
    )


def get_sleep_loss(generative_model, inference_network, num_samples=1):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """

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
