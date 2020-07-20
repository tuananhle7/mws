import torch
import losses
import util
import itertools
import models


def train_mws(generative_model, inference_network, data_loader,
              num_iterations, memory_size, true_cluster_cov,
              test_data_loader, test_num_particles, true_generative_model,
              checkpoint_path, reweighted=False):
    optimizer = torch.optim.Adam(itertools.chain(
        generative_model.parameters(), inference_network.parameters()))
    (theta_losses, phi_losses, cluster_cov_distances,
     test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
     train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
     train_kl_memory_ps, train_kl_memory_ps_true) = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    memory = {}
    data_loader_iter = iter(data_loader)

    for iteration in range(num_iterations):
        # get obs
        try:
            obs = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            obs = next(data_loader_iter)

        theta_loss = 0
        phi_loss = 0
        for single_obs in obs:
            # key to index memory
            single_obs_key = tuple(single_obs.tolist())

            # populate memory if empty
            if (
                (single_obs_key not in memory) or
                len(memory[single_obs_key]) == 0
            ):
                # batch shape [1] and event shape [num_data]
                latent_dist = inference_network.get_latent_dist(
                    single_obs.unsqueeze(0))

                # HACK
                while True:
                    # [memory_size, num_data]
                    latent = inference_network.sample_from_latent_dist(
                        latent_dist, memory_size).squeeze(1)
                    # list of M \in {1, ..., memory_size} elements
                    # could be less than memory_size because
                    # sampled elements can be duplicate
                    memory[single_obs_key] = list(set(
                        [tuple(x.tolist()) for x in latent]))

                    if len(memory[single_obs_key]) == memory_size:
                        break

            # WAKE
            # batch shape [1] and event shape [num_data]
            latent_dist = inference_network.get_latent_dist(
                single_obs.unsqueeze(0))
            # [1, 1, num_data] -> [num_data]
            latent = inference_network.sample_from_latent_dist(
                latent_dist, 1).view(-1)
            # set (of size memory_size + 1) of tuples (of length num_data)
            memoized_latent_plus_current_latent = set(
                memory.get(single_obs_key, []) +
                [tuple(latent.tolist())]
            )

            # [memory_size + 1, 1, num_data]
            memoized_latent_plus_current_latent_tensor = torch.tensor(
                list(memoized_latent_plus_current_latent),
                device=single_obs.device
            ).unsqueeze(1)
            # [memory_size + 1]
            log_p_tensor = generative_model.get_log_prob(
                memoized_latent_plus_current_latent_tensor,
                single_obs.unsqueeze(0)
            ).squeeze(-1)

            # this takes the longest
            # {int: [], ...}
            log_p = {mem_latent: lp for mem_latent, lp in zip(
                memoized_latent_plus_current_latent, log_p_tensor)}

            # update memory.
            # {float: list of ints}
            memory[single_obs_key] = sorted(
                memoized_latent_plus_current_latent,
                key=log_p.get)[-memory_size:]

            # REMEMBER
            # []
            if reweighted:
                memory_log_weight = torch.stack(
                    list(map(log_p.get, memory[single_obs_key])))  # [memory_size]
                memory_weight_normalized = util.exponentiate_and_normalize(memory_log_weight, dim=0)  # [memory_size]
                memory_latent = torch.tensor(memory[single_obs_key])  # [memory_size, num_data]
                inference_network_log_prob = inference_network.get_log_prob_from_latent_dist(
                    latent_dist, memory_latent[:, None, :]).squeeze(-1)  # [memory_size]

                theta_loss += -torch.sum(memory_log_weight * memory_weight_normalized.detach()) / len(obs)
                phi_loss += -torch.sum(inference_network_log_prob * memory_weight_normalized.detach()) / len(obs)
            else:
                remembered_latent_id_dist = torch.distributions.Categorical(
                    logits=torch.tensor(
                        list(map(log_p.get, memory[single_obs_key])))
                )
                remembered_latent_id = remembered_latent_id_dist.sample()
                remembered_latent_id_log_prob = remembered_latent_id_dist.log_prob(
                    remembered_latent_id)
                remembered_latent = memory[single_obs_key][remembered_latent_id]
                remembered_latent_tensor = torch.tensor(
                    [remembered_latent],
                    device=single_obs.device)
                # []
                theta_loss += -(log_p.get(remembered_latent) -
                                remembered_latent_id_log_prob.detach()) / len(obs)
                # []
                phi_loss += -inference_network.get_log_prob_from_latent_dist(
                    latent_dist, remembered_latent_tensor).view(()) / len(obs)

            # SLEEP
            # TODO

        optimizer.zero_grad()
        theta_loss.backward()
        phi_loss.backward()
        optimizer.step()

        theta_losses.append(theta_loss.item())
        phi_losses.append(phi_loss.item())
        cluster_cov_distances.append(torch.norm(
            true_cluster_cov - generative_model.get_cluster_cov()
        ).item())

        if iteration % 100 == 0:  # test every 100 iterations
            (test_log_p, test_log_p_true,
             test_kl_qp, test_kl_pq, test_kl_qp_true, test_kl_pq_true,
             _, _) = models.eval_gen_inf(
                true_generative_model, generative_model, inference_network, None,
                test_data_loader)
            test_log_ps.append(test_log_p)
            test_log_ps_true.append(test_log_p_true)
            test_kl_qps.append(test_kl_qp)
            test_kl_pqs.append(test_kl_pq)
            test_kl_qps_true.append(test_kl_qp_true)
            test_kl_pqs_true.append(test_kl_pq_true)

            (train_log_p, train_log_p_true,
             train_kl_qp, train_kl_pq, train_kl_qp_true, train_kl_pq_true,
             train_kl_memory_p, train_kl_memory_p_true) = models.eval_gen_inf(
                true_generative_model, generative_model, inference_network, memory,
                data_loader)
            train_log_ps.append(train_log_p)
            train_log_ps_true.append(train_log_p_true)
            train_kl_qps.append(train_kl_qp)
            train_kl_pqs.append(train_kl_pq)
            train_kl_qps_true.append(train_kl_qp_true)
            train_kl_pqs_true.append(train_kl_pq_true)
            train_kl_memory_ps.append(train_kl_memory_p)
            train_kl_memory_ps_true.append(train_kl_memory_p_true)

            util.save_checkpoint(
                checkpoint_path, generative_model, inference_network,
                theta_losses, phi_losses, cluster_cov_distances,
                test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
                train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
                train_kl_memory_ps, train_kl_memory_ps_true, memory, None, None)

        util.print_with_time(
            'it. {} | theta loss = {:.2f} | phi loss = {:.2f}'.format(
                iteration, theta_loss, phi_loss))

        # if iteration % 200 == 0:
        #     z = inference_network.get_latent_dist(obs).sample()
        #     util.save_plot("images/mws/iteration_{}.png".format(iteration),
        #                    obs[:3], z[:3])

    return (theta_losses, phi_losses, cluster_cov_distances,
            test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
            train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
            train_kl_memory_ps, train_kl_memory_ps_true, memory, None, None)


def train_rws(generative_model, inference_network, data_loader,
              num_iterations, num_particles, true_cluster_cov,
              test_data_loader, test_num_particles, true_generative_model,
              checkpoint_path):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    optimizer_theta = torch.optim.Adam(generative_model.parameters())
    (theta_losses, phi_losses, cluster_cov_distances,
     test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
     train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
     reweighted_train_kl_qps, reweighted_train_kl_qps_true) = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    data_loader_iter = iter(data_loader)

    for iteration in range(num_iterations):
        # get obs
        try:
            obs = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            obs = next(data_loader_iter)

        log_weight, log_q = losses.get_log_weight_and_log_q(
            generative_model, inference_network, obs, num_particles)

        # wake theta
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
            log_weight)
        wake_theta_loss.backward(retain_graph=True)
        optimizer_theta.step()

        # wake phi
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
            log_weight, log_q)
        wake_phi_loss.backward()
        optimizer_phi.step()

        theta_losses.append(wake_theta_loss.item())
        phi_losses.append(wake_phi_loss.item())
        cluster_cov_distances.append(torch.norm(
            true_cluster_cov - generative_model.get_cluster_cov()
        ).item())
        if iteration % 100 == 0:  # test every 100 iterations
            (test_log_p, test_log_p_true,
             test_kl_qp, test_kl_pq, test_kl_qp_true, test_kl_pq_true,
             _, _) = models.eval_gen_inf(
                true_generative_model, generative_model, inference_network, None,
                test_data_loader)
            test_log_ps.append(test_log_p)
            test_log_ps_true.append(test_log_p_true)
            test_kl_qps.append(test_kl_qp)
            test_kl_pqs.append(test_kl_pq)
            test_kl_qps_true.append(test_kl_qp_true)
            test_kl_pqs_true.append(test_kl_pq_true)

            (train_log_p, train_log_p_true,
             train_kl_qp, train_kl_pq, train_kl_qp_true, train_kl_pq_true,
             _, _, reweighted_train_kl_qp, reweighted_train_kl_qp_true) = models.eval_gen_inf(
                true_generative_model, generative_model, inference_network, None,
                data_loader, num_particles=num_particles, reweighted_kl=True)
            train_log_ps.append(train_log_p)
            train_log_ps_true.append(train_log_p_true)
            train_kl_qps.append(train_kl_qp)
            train_kl_pqs.append(train_kl_pq)
            train_kl_qps_true.append(train_kl_qp_true)
            train_kl_pqs_true.append(train_kl_pq_true)
            reweighted_train_kl_qps.append(reweighted_train_kl_qp)
            reweighted_train_kl_qps_true.append(reweighted_train_kl_qp_true)

            util.save_checkpoint(
                checkpoint_path, generative_model, inference_network,
                theta_losses, phi_losses, cluster_cov_distances,
                test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
                train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
                None, None, None, reweighted_train_kl_qps, reweighted_train_kl_qps_true)

        util.print_with_time(
            'it. {} | theta loss = {:.2f} | phi loss = {:.2f}'.format(
                iteration, wake_theta_loss, wake_phi_loss))

        # if iteration % 200 == 0:
        #     z = inference_network.get_latent_dist(obs).sample()
        #     util.save_plot("images/rws/iteration_{}.png".format(iteration),
        #                    obs[:3], z[:3])

    return (theta_losses, phi_losses, cluster_cov_distances,
            test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
            train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
            None, None, None, reweighted_train_kl_qps, reweighted_train_kl_qps_true)


def train_vimco(generative_model, inference_network, data_loader,
                num_iterations, num_particles, true_cluster_cov,
                test_data_loader, test_num_particles, true_generative_model,
                checkpoint_path):
    optimizer = torch.optim.Adam(itertools.chain(
        generative_model.parameters(), inference_network.parameters()))
    (theta_losses, phi_losses, cluster_cov_distances,
     test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
     train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
     reweighted_train_kl_qps, reweighted_train_kl_qps_true) = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    data_loader_iter = iter(data_loader)

    for iteration in range(num_iterations):
        # get obs
        try:
            obs = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            obs = next(data_loader_iter)

        # loss
        optimizer.zero_grad()
        loss, elbo = losses.get_vimco_loss(
            generative_model, inference_network, obs, num_particles)
        loss.backward(retain_graph=True)
        optimizer.step()

        theta_losses.append(loss.item())
        phi_losses.append(loss.item())
        cluster_cov_distances.append(torch.norm(
            true_cluster_cov - generative_model.get_cluster_cov()
        ).item())
        if iteration % 100 == 0:  # test every 100 iterations
            (test_log_p, test_log_p_true,
             test_kl_qp, test_kl_pq, test_kl_qp_true, test_kl_pq_true,
             _, _) = models.eval_gen_inf(
                true_generative_model, generative_model, inference_network, None,
                test_data_loader)
            test_log_ps.append(test_log_p)
            test_log_ps_true.append(test_log_p_true)
            test_kl_qps.append(test_kl_qp)
            test_kl_pqs.append(test_kl_pq)
            test_kl_qps_true.append(test_kl_qp_true)
            test_kl_pqs_true.append(test_kl_pq_true)

            (train_log_p, train_log_p_true,
             train_kl_qp, train_kl_pq, train_kl_qp_true, train_kl_pq_true,
             _, _, reweighted_train_kl_qp, reweighted_train_kl_qp_true) = models.eval_gen_inf(
                true_generative_model, generative_model, inference_network, None,
                data_loader, num_particles=num_particles, reweighted_kl=True)
            train_log_ps.append(train_log_p)
            train_log_ps_true.append(train_log_p_true)
            train_kl_qps.append(train_kl_qp)
            train_kl_pqs.append(train_kl_pq)
            train_kl_qps_true.append(train_kl_qp_true)
            train_kl_pqs_true.append(train_kl_pq_true)
            reweighted_train_kl_qps.append(reweighted_train_kl_qp)
            reweighted_train_kl_qps_true.append(reweighted_train_kl_qp_true)

            util.save_checkpoint(
                checkpoint_path, generative_model, inference_network,
                theta_losses, phi_losses, cluster_cov_distances,
                test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
                train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
                None, None, None, reweighted_train_kl_qps, reweighted_train_kl_qps_true)

        util.print_with_time(
            'it. {} | theta loss = {:.2f}'.format(iteration, loss))

        # if iteration % 200 == 0:
        #     z = inference_network.get_latent_dist(obs).sample()
        #     util.save_plot("images/rws/iteration_{}.png".format(iteration),
        #                    obs[:3], z[:3])

    return (theta_losses, phi_losses, cluster_cov_distances,
            test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
            train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true, train_kl_pqs_true,
            None, None, None, reweighted_train_kl_qps, reweighted_train_kl_qps_true)
