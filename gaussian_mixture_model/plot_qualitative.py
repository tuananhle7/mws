import util
import torch
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os
import scipy


train_data, test_data = util.load_data()


# def get_predictive_obs(num_samples, obs, inference_network, generative_model):
#     batch_size = obs.shape[0]
#     inference_network_dist = inference_network.get_latent_dist(obs)
#     latent = inference_network_dist.sample()
#     obs_dist = generative_model.get_obs_dist(latent)
#     return latent, obs_dist.sample(sample_shape=(num_samples,)).permute(1, 0, 2).reshape(batch_size, -1, 2)


# def get_predictive_obs_from_memory(num_samples, obs, memory, generative_model):
#     latent = []
#     predictive_obs = []
#     for single_obs in obs:
#         single_obs_key = tuple(single_obs.tolist())
#         memory_latent = torch.tensor(memory[single_obs_key])
#         memory_log_prob_unnormalized = generative_model.get_log_prob(
#             memory_latent[:, None, :], single_obs[None])[:, 0]
#         latent.append(memory_latent[torch.distributions.Categorical(logits=memory_log_prob_unnormalized).sample()])
#         obs_dist = generative_model.get_obs_dist(latent[-1][None])
#         predictive_obs.append(obs_dist.sample(sample_shape=(num_samples,))[:, 0, :].reshape(-1, 2))
#     return torch.stack(latent), torch.stack(predictive_obs)


def get_predictive_obs(num_samples_per_something, obs, inference_network, generative_model):
    obs = train_data
    batch_size, obs_dim = obs.shape
    num_data = int(obs_dim / 2)
    inference_network_dist = inference_network.get_latent_dist(obs)
    latent = inference_network_dist.sample()

    obs_precision = generative_model.get_cluster_cov().detach().inverse()
    prior_precision = generative_model.prior_cov.inverse()
    prior_loc = generative_model.prior_loc

    predictive_obs = []
    for single_obs, single_latent in zip(obs, latent):
        temp = []
        for c in range(num_data):
            obs_in_cluster_c = get_obs_in_cluster_c(single_obs.reshape(-1, 2), single_latent, c)
            num_data_c = len(obs_in_cluster_c)
            if num_data_c > 0:
                precision = prior_precision + obs_precision * num_data_c
                cov = precision.inverse()
                loc = torch.mv(
                    cov,
                    torch.mv(prior_precision, prior_loc)
                    + torch.einsum("ab,nb->a", obs_precision, obs_in_cluster_c),
                )
                temp.append(
                    torch.distributions.MultivariateNormal(loc, cov).sample(
                        (int(num_samples_per_something * num_data_c),)
                    )
                )
        predictive_obs.append(torch.cat(temp, dim=0))

    return latent, torch.stack(predictive_obs)


def get_predictive_obs_from_memory(num_samples_per_something, obs, memory, generative_model):
    obs = train_data
    batch_size, obs_dim = obs.shape
    num_data = int(obs_dim / 2)

    obs_precision = generative_model.get_cluster_cov().detach().inverse()
    prior_precision = generative_model.prior_cov.inverse()
    prior_loc = generative_model.prior_loc

    latent = []
    predictive_obs = []
    for single_obs in obs:
        single_obs_key = tuple(single_obs.tolist())
        memory_latent = torch.tensor(memory[single_obs_key])
        memory_log_prob_unnormalized = generative_model.get_log_prob(
            memory_latent[:, None, :], single_obs[None]
        )[:, 0]
        single_latent = memory_latent[
            torch.distributions.Categorical(logits=memory_log_prob_unnormalized).sample()
        ]
        latent.append(single_latent)

        temp = []
        for c in range(num_data):
            obs_in_cluster_c = get_obs_in_cluster_c(single_obs.reshape(-1, 2), single_latent, c)
            num_data_c = len(obs_in_cluster_c)
            if num_data_c > 0:
                precision = prior_precision + obs_precision * num_data_c
                cov = precision.inverse()
                loc = torch.mv(
                    cov,
                    torch.mv(prior_precision, prior_loc)
                    + torch.einsum("ab,nb->a", obs_precision, obs_in_cluster_c),
                )
                temp.append(
                    torch.distributions.MultivariateNormal(loc, cov).sample(
                        (int(num_samples_per_something * num_data_c),)
                    )
                )
        predictive_obs.append(torch.cat(temp, dim=0))
    return latent, torch.stack(predictive_obs)


def get_obs_in_cluster_c(obs, latent, c):
    return obs[latent == c]


def plot_clustering(ax, obs, latent):
    for c in range(num_data):
        obs_in_cluster_c = get_obs_in_cluster_c(obs, latent, c)
        ax.plot(
            obs_in_cluster_c[:, 0],
            obs_in_cluster_c[:, 1],
            linestyle="",
            marker="o",
            color=colors[c],
        )


def plot_kde(ax, predictive_obs):
    xmin, xmax = -5, 5
    ymin, ymax = -5, 5
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = predictive_obs.t().numpy()
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X, Y, Z, colors="grey")
    return ax


def load(algorithm, seed, num_particles):
    checkpoint_path = "checkpoints/checkpoint_{}_{}_{}.pt".format(algorithm, seed, num_particles)

    (
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
        _,
        _,
    ) = util.load_checkpoint(checkpoint_path, torch.device("cpu"))

    return generative_model, inference_network, memory


def plot_normal2d(ax, mean, cov, num_points=100, confidence=0.95, **kwargs):
    # https://stats.stackexchange.com/questions/64680/how-to-determine-quantiles-isolines-of-a-multivariate-normal-distribution
    # plots a `confidence' probability ellipse
    const = -2 * np.log(1 - confidence)
    eigvals, eigvecs = scipy.linalg.eig(np.linalg.inv(cov))
    eigvals = np.real(eigvals)
    a = np.sqrt(const / eigvals[0])
    b = np.sqrt(const / eigvals[1])
    theta = np.linspace(-np.pi, np.pi, num=num_points)
    xy = eigvecs @ np.array([np.cos(theta) * a, np.sin(theta) * b]) + np.expand_dims(mean, -1)
    ax.plot(xy[0, :], xy[1, :], **kwargs)
    return ax


if __name__ == "__main__":
    train_data, test_data = util.load_data()
    num_samples_per_something = 100
    num_particless = [2, 5, 10, 20, 50]
    seed = 1
    num_datasets = 10
    algorithms = ["mws", "rws", "vimco"]
    num_data = 7
    colors = ["C{}".format(i) for i in range(num_data)]

    for num_particles in num_particless:
        fig, axss = plt.subplots(
            num_datasets,
            len(algorithms),
            sharex=True,
            sharey=True,
            figsize=(len(algorithms) * 3, num_datasets * 3),
            dpi=100,
        )

        for j, algorithm in enumerate(algorithms):
            generative_model, inference_network, memory = load(algorithm, seed, num_particles)

            if algorithm == "mws":
                latent, predictive_obs = get_predictive_obs_from_memory(
                    num_samples_per_something, train_data, memory, generative_model
                )
            else:
                latent, predictive_obs = get_predictive_obs(
                    num_samples_per_something, train_data, inference_network, generative_model
                )

            for i, obs in enumerate(train_data[:num_datasets]):
                ax = axss[i, j]
                ax.plot(
                    predictive_obs[i][:, 0],
                    predictive_obs[i][:, 1],
                    marker=".",
                    color="black",
                    alpha=0.1,
                    markeredgewidth=0,
                    linestyle="",
                    markersize=5,
                )
                plot_clustering(ax, train_data[i].reshape(-1, 2), latent[i])
        #         plot_kde(ax, predictive_obs[i])

        for algorithm, ax in zip(algorithms, axss[0]):
            ax.set_title(algorithm.upper())

        for axs in axss:
            for ax in axs:
                ax.set_xlim(-3, 2)
                ax.set_ylim(-3, 4)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.tight_layout(pad=0)
        path = os.path.join("diagnostics/qualitative_{}.png".format(num_particles))
        fig.savefig(path, bbox_inches="tight")
        print("Saved to {}".format(path))
        plt.close(fig)

    seed = 1
    fig, axs = plt.subplots(
        1, len(num_particless), sharex=True, sharey=True, figsize=(3 * len(num_particless), 3)
    )

    for num_particles, ax in zip(num_particless, axs):
        ax.set_title("K = {}".format(num_particles))
        plot_normal2d(ax, [0, 0], np.eye(2) * 0.03, color="black", label="True")
        for algorithm, color in zip(algorithms, colors):
            cluster_cov = load(algorithm, seed, num_particles)[0].get_cluster_cov().detach().numpy()
            plot_normal2d(ax, [0, 0], cluster_cov, color=color, label=algorithm.upper())

    axs[-1].legend(ncol=2)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0)
    path = os.path.join("diagnostics/cluster_cov.png")
    fig.savefig(path, bbox_inches="tight")
    print("Saved to {}".format(path))
    plt.close(fig)
