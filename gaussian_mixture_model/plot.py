import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D

import util


def moving_average(x, width=100):
    return np.concatenate(
        [np.full(width - 1, np.nan), np.convolve(x, np.ones(width) / width, "valid")]
    )


def plot_errors(ax, mid, lower, upper, **plot_kwargs):
    xs = np.arange(len(mid))
    ax.plot(xs, mid, **plot_kwargs)
    ax.fill_between(xs, lower, upper, alpha=0.2, **plot_kwargs)


def plot_errors_end(ax, num_particles, mid, lower, upper, **plot_kwargs):
    ax.errorbar(
        num_particles,
        mid[-1],
        yerr=[[mid[-1] - lower[-1]], [upper[-1] - mid[-1]]],
        fmt="o",
        **plot_kwargs
    )
    # ax.fill_between(xs, lower, upper, alpha=0.2, **plot_kwargs)


def load(algorithm, num_particles):
    things = [[] for _ in range(19)]
    # for seed in range(1, 11):
    for seed in range(3):
        checkpoint_path = "checkpoints/checkpoint_{}_{}_{}.pt".format(
            algorithm, num_particles, seed
        )
        checkpoint = util.load_checkpoint(checkpoint_path, torch.device("cpu"))
        for i, x in enumerate(checkpoint[2:-3] + checkpoint[-2:]):
            if i >= 17 and (algorithm == "rws" or algorithm == "vimco"):
                things[i].append(moving_average(x, 10))
            else:
                things[i].append(x)

    # cut to min length
    for i in range(len(things)):
        if i < 2:
            things[i] = [moving_average(x) for x in things[i]]
        if things[i][0] is not None:
            min_length = min(map(len, things[i]))
            things[i] = [x[:min_length] for x in things[i]]

    mid, lower, upper = [], [], []
    for i, thing in enumerate(things):
        if thing[0] is None:
            mid.append(None)
            lower.append(None)
            upper.append(None)
        else:
            mid.append(np.quantile(np.array(thing), 0.5, axis=0))
            lower.append(np.quantile(np.array(thing), 0.25, axis=0))
            upper.append(np.quantile(np.array(thing), 0.75, axis=0))

    # mid = [np.quantile(np.array(thing), 0.5, axis=0) for thing in things]
    # lower = [np.quantile(np.array(thing), 0.25, axis=0) for thing in things]
    # upper = [np.quantile(np.array(thing), 0.75, axis=0) for thing in things]
    return mid, lower, upper


def get_predictive_obs(num_samples_per_something, obs, inference_network, generative_model):
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
    num_data = obs.shape[0]
    colors = ["C{}".format(i) for i in range(num_data)]
    for c in range(num_data):
        obs_in_cluster_c = get_obs_in_cluster_c(obs, latent, c)
        ax.plot(
            obs_in_cluster_c[:, 0],
            obs_in_cluster_c[:, 1],
            linestyle="",
            marker="o",
            color=colors[c],
            markeredgewidth=0.2,
            markeredgecolor="black",
        )


def load_gen_inf_mem(algorithm, num_particles, seed):
    checkpoint_path = "checkpoints/checkpoint_{}_{}_{}.pt".format(algorithm, num_particles, seed)

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


def main(args):
    util.set_seed(1)

    if not os.path.exists(args.diagnostics_dir):
        os.makedirs(args.diagnostics_dir)

    num_particless = [2, 5, 10, 20, 50]
    # num_particles_detailed = 10
    # colors = ['C0', 'C1', 'C2', 'C5']
    # algorithms = ['mws', 'rws', 'vimco', 'rmws']
    colors = ["C0", "C1", "C2"]
    linestyles = ["--", "-", ":"]
    # algorithms = ['mws', 'rws', 'vimco']
    algorithms = ["rws", "vimco", "mws"]
    fig, axss = plt.subplots(2, 4, figsize=(4 * 3, 2 * 3), dpi=100)
    for num_particles_id, num_particles in enumerate(num_particless):
        for color, algorithm in zip(colors, algorithms):
            mid, lower, upper = load(algorithm, num_particles)

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
                reweighted_train_kl_qps,
                reweighted_train_kl_qps_true,
            ) = zip(mid, lower, upper)

            Ks = [2, 20]
            if num_particles in Ks:
                ax = axss[0, 0]
                plot_errors(
                    ax,
                    *[-train_log_p for train_log_p in train_log_ps],
                    color=color,
                    linestyle=linestyles[Ks.index(num_particles)],
                    zorder=1
                )

                ax = axss[0, 1]
                if algorithm == "mws" or algorithm == "rmws":
                    plot_errors(
                        ax,
                        *train_kl_memory_ps_true,
                        color=color,
                        linestyle=linestyles[Ks.index(num_particles)]
                    )
                else:
                    plot_errors(
                        ax,
                        *reweighted_train_kl_qps_true,
                        color=color,
                        linestyle=linestyles[Ks.index(num_particles)]
                    )

            ax = axss[0, 2]
            plot_errors_end(
                ax,
                num_particles_id,
                *[-train_log_p for train_log_p in train_log_ps],
                color=color,
                clip_on=False,
                zorder=10
            )

            ax = axss[0, 3]
            if algorithm == "mws" or algorithm == "rmws":
                plot_errors_end(
                    ax,
                    num_particles_id,
                    *train_kl_memory_ps_true,
                    color=color,
                    clip_on=False,
                    zorder=10
                )
            else:
                plot_errors_end(
                    ax,
                    num_particles_id,
                    *reweighted_train_kl_qps_true,
                    color=color,
                    clip_on=False,
                    zorder=10
                )

    ax = axss[0, 0]
    ax.set_ylim(-train_log_ps_true[0][0], 16)
    # ax.set_yticks([-train_log_ps_true[0][0]])
    # ax.set_yticklabels(['True'], rotation='vertical', verticalalignment='center')
    ax.set_yticks([])
    # ax.text(0.01, 0.01, 'True',
    #         horizontalalignment='center',
    #         verticalalignment='bottom',
    #         transform=ax.transAxes)
    ax.axhline(-train_log_ps_true[0][0], color="black", zorder=0)
    ax.set_xticks([0, 499])
    ax.set_xticklabels(["", "50k"])
    ax.xaxis.set_label_coords(0.5, -0.03)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("-log p")
    ax.set_xticks([0, len(train_log_ps[0]) - 1])
    ax.yaxis.set_label_coords(0, 0.5)
    sns.despine(ax=ax, left=True)

    ax = axss[0, 1]
    ax.set_ylim(0, 35)
    # ax.set_yticks([0])
    ax.set_yticks([])
    ax.axhline(0, color="black")
    ax.set_xticks([0, 499])
    ax.set_xticklabels(["", "50k"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL(q || p)")
    ax.set_xticks([0, len(train_kl_qps_true[0]) - 1])
    ax.yaxis.set_label_coords(0, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.03)
    sns.despine(ax=ax, left=True)

    ax12 = fig.add_subplot(221, frameon=False)
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax12.set_title("Speed of learning")

    ax = axss[0, 2]
    ax.set_ylim(-train_log_ps_true[0][0], 16)
    ax.axhline(-train_log_ps_true[0][0], color="black")
    # ax.set_yticks([-train_log_ps_true[0][0]])
    # ax.set_yticklabels(['True'], rotation='vertical', verticalalignment='center')
    ax.set_yticks([])
    # ax.text(0.01, 0.01, 'True',
    #         horizontalalignment='center',
    #         verticalalignment='bottom',
    #         transform=ax.transAxes)
    ax.set_ylabel("-log p")
    ax.set_xticks(np.arange(len(num_particless)))
    ax.set_xticklabels(num_particless)
    ax.set_xlabel("K")
    # ax.xaxis.set_label_coords(0.5, -0.03)
    ax.yaxis.set_label_coords(0, 0.5)
    sns.despine(ax=ax, left=True)

    ax = axss[0, 3]
    ax.set_ylim(0, 35)
    # ax.set_yticks([0])
    ax.set_yticks([])
    ax.axhline(0, color="black")
    ax.set_ylabel("KL(q || p)")
    ax.set_xticks(np.arange(len(num_particless)))
    ax.set_xticklabels(num_particless)
    ax.set_xlabel("K")
    # ax.xaxis.set_label_coords(0.5, -0.03)
    ax.yaxis.set_label_coords(0, 0.5)
    sns.despine(ax=ax, left=True)

    ax34 = fig.add_subplot(222, frameon=False)
    ax34.set_xticks([])
    ax34.set_yticks([])
    ax34.set_title("Quality of the learned model and inference")

    order = [0, 1, 2]
    lines = [
        # Line2D([0], [0], color=colors[order[0]]),
        # Line2D([0], [0], color=colors[order[1]]),
        # Line2D([0], [0], color=colors[order[2]]),
        Line2D([0], [0], color="grey", linestyle="--"),
        Line2D([0], [0], color="grey", linestyle="-"),
    ]
    labels = [
        # algorithms[order[0]].upper(),
        # algorithms[order[1]].upper(),
        # algorithms[order[2]].upper(),
        "K = 2",
        "K = 20",
        # 'MWS (K = {})'.format(num_particles_detailed),
        # 'RWS (K = {})'.format(num_particles_detailed),
        # 'VIMCO (K = {})'.format(num_particles_detailed),
    ]
    axss[0, 1].legend(lines, labels, loc="upper right")

    lines = [
        Line2D([0], [0], color=colors[order[0]], marker="o", linestyle="-"),
        Line2D([0], [0], color=colors[order[1]], marker="o", linestyle="-"),
        Line2D([0], [0], color=colors[order[2]], marker="o", linestyle="-"),
        Line2D([0], [0], color="black", linestyle="-"),
    ]
    labels = [
        algorithms[order[0]].upper(),
        algorithms[order[1]].upper(),
        algorithms[order[2]].upper(),
        "Ground truth",
    ]
    axss[0, 3].legend(lines, labels)

    for ax in axss[0]:
        ax.tick_params(length=0)

    num_samples_per_something = 100
    num_particles = 5
    seed = 1
    i = 8
    train_data, test_data = util.load_data()

    ax = axss[1, 0]
    # ax.set_title('Data')
    ax.text(
        0.98,
        0.98,
        "Data",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    obs = train_data[i].reshape(-1, 2)
    ax.plot(
        obs[:, 0],
        obs[:, 1],
        linestyle="",
        marker="o",
        color="black",
        markeredgewidth=0.2,
        markeredgecolor="white",
    )

    for j in range(len(algorithms)):
        algorithm = algorithms[order[j]]
        generative_model, inference_network, memory = load_gen_inf_mem(
            algorithm, num_particles, seed
        )

        if algorithm == "mws":
            latent, predictive_obs = get_predictive_obs_from_memory(
                num_samples_per_something, train_data, memory, generative_model
            )
        else:
            latent, predictive_obs = get_predictive_obs(
                num_samples_per_something, train_data, inference_network, generative_model
            )

        ax = axss[1, j + 1]
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
        # ax.set_title(algorithm.upper())
        ax.text(
            0.98,
            0.98,
            algorithm.upper(),
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    for ax in axss[1]:
        ax.set_xlim(-2, 0.75)
        ax.set_ylim(-1, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])

    ax5678 = fig.add_subplot(212, frameon=False)
    ax5678.set_xticks([])
    ax5678.set_yticks([])
    ax5678.set_title("Clustering and posterior predictive")

    fig.tight_layout(h_pad=0.5, w_pad=0.2)
    path = os.path.join(args.diagnostics_dir, "overall_plot.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("Saved to {}".format(path))
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-path-prefix", default="checkpoint", help=" ")
    parser.add_argument("--diagnostics-dir", default="diagnostics/", help=" ")
    args = parser.parse_args()
    main(args)
