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
    for seed in range(1, 11):
        checkpoint_path = "checkpoints/checkpoint_{}_{}_{}.pt".format(
            algorithm, seed, num_particles
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


def main(args):
    if not os.path.exists(args.diagnostics_dir):
        os.makedirs(args.diagnostics_dir)

    num_particless = [2, 5, 10, 20, 50]
    # colors = ['C0', 'C1', 'C2', 'C5']
    # algorithms = ['mws', 'rws', 'vimco', 'rmws']
    colors = ["C0", "C1", "C2"]
    algorithms = ["mws", "rws", "vimco"]
    for num_particles in num_particless:
        fig, axss = plt.subplots(3, 5, figsize=(5 * 3, 3 * 3), dpi=100)
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

            # import pdb; pdb.set_trace()
            ax = axss[0, 0]
            plot_errors(ax, *theta_losses, color=color)
            # ax.plot(moving_average(theta_losses), label=algorithm)
            ax.set_xlabel("iteration")
            ax.set_ylabel("model loss")
            ax.set_xticks([0, len(theta_losses[0]) - 1])

            ax = axss[0, 1]
            plot_errors(ax, *phi_losses, color=color)
            # ax.plot(moving_average(phi_losses), label=algorithm)
            ax.set_xlabel("iteration")
            ax.set_ylabel("inference loss")
            ax.set_xticks([0, len(phi_losses[0]) - 1])

            ax = axss[0, 2]
            plot_errors(ax, *cluster_cov_distances, color=color)
            # ax.plot(cluster_cov_distances, label=algorithm)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("cluster cov distance")
            ax.set_xticks([0, len(cluster_cov_distances[0]) - 1])

            for ax in axss[0, 3:]:
                ax.axis("off")

            ax = axss[1, 0]
            plot_errors(ax, *test_log_ps, color=color)
            # lines = ax.plot(test_log_ps, label=algorithm)
            ax.plot(test_log_ps_true[0], color="black")
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("TEST\nlog p")
            ax.set_xticks([0, len(test_log_ps[0]) - 1])

            ax = axss[1, 1]
            plot_errors(ax, *test_kl_qps, color=color)
            # ax.plot(test_kl_qps, label=algorithm)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(q, p)")
            ax.set_xticks([0, len(test_kl_qps[0]) - 1])

            ax = axss[1, 2]
            plot_errors(ax, *test_kl_pqs, color=color)
            # ax.plot(test_kl_pqs, label=algorithm)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(p, q)")
            ax.set_xticks([0, len(test_kl_pqs[0]) - 1])

            ax = axss[1, 3]
            plot_errors(ax, *test_kl_qps_true, color=color)
            # ax.plot(test_kl_qps_true, label=algorithm)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(q, p true)")
            ax.set_xticks([0, len(test_kl_qps_true[0]) - 1])

            ax = axss[1, 4]
            plot_errors(ax, *test_kl_pqs_true, color=color)
            # ax.plot(test_kl_pqs_true, label=algorithm)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(p true, q)")
            ax.set_xticks([0, len(test_kl_pqs_true[0]) - 1])

            ax = axss[2, 0]
            plot_errors(ax, *train_log_ps, color=color)
            # lines = ax.plot(train_log_ps, label=algorithm)
            ax.plot(train_log_ps_true[1], color="black")
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("TRAIN\nlog p")
            ax.set_xticks([0, len(train_log_ps[0]) - 1])

            ax = axss[2, 1]
            plot_errors(ax, *train_kl_qps, color=color)
            # lines = ax.plot(train_kl_qps, label=algorithm)
            if algorithm == "mws" or algorithm == "rmws":
                plot_errors(ax, *train_kl_memory_ps, linestyle="dashed", color=color)
                # ax.plot(train_kl_memory_ps, label=algorithm,
                # color=lines[0].get_color(), linestyle='dashed')
            else:
                plot_errors(ax, *reweighted_train_kl_qps, linestyle="dashed", color=color)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(q, p)")
            ax.set_xticks([0, len(train_kl_qps[0]) - 1])

            ax = axss[2, 2]
            plot_errors(ax, *train_kl_pqs, color=color)
            # ax.plot(train_kl_pqs, label=algorithm)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(p, q)")
            ax.set_xticks([0, len(train_kl_pqs[0]) - 1])

            ax = axss[2, 3]
            plot_errors(ax, *train_kl_qps_true, color=color)
            # lines = ax.plot(train_kl_qps_true, label=algorithm)
            if algorithm == "mws" or algorithm == "rmws":
                plot_errors(ax, *train_kl_memory_ps_true, linestyle="dashed", color=color)
                # ax.plot(train_kl_memory_ps_true, label=algorithm,
                # color=lines[0].get_color(), linestyle='dashed')
            else:
                plot_errors(ax, *reweighted_train_kl_qps_true, linestyle="dashed", color=color)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(q, p true)")
            ax.set_xticks([0, len(train_kl_qps_true[0]) - 1])

            ax = axss[2, 4]
            plot_errors(ax, *train_kl_pqs_true, color=color)
            # ax.plot(train_kl_pqs_true, label=algorithm)
            ax.set_xlabel("iteration / 100")
            ax.set_ylabel("KL(p true, q)")
            ax.set_xticks([0, len(train_kl_pqs_true[0]) - 1])

        lines = [
            Line2D([0], [0], color=colors[0]),
            Line2D([0], [0], color=colors[0], linestyle="dashed"),
            Line2D([0], [0], color=colors[1]),
            Line2D([0], [0], color=colors[2]),
            # Line2D([0], [0], color=colors[3]),
            # Line2D([0], [0], color=colors[3], linestyle='dashed'),
        ]
        labels = [
            "MWS",
            "MWS memory",
            "RWS",
            "VIMCO",
            # 'R-MWS',
            # 'R-MWS memory',
        ]
        axss[0, 0].legend(lines, labels)

        for axs in axss:
            for ax in axs:
                sns.despine(ax=ax)
                ax.xaxis.set_label_coords(0.5, -0.01)
                ax.tick_params(direction="in")

        fig.tight_layout(pad=0)
        path = os.path.join(args.diagnostics_dir, "losses_{}.png".format(num_particles))
        fig.savefig(path, bbox_inches="tight")
        print("Saved to {}".format(path))
        plt.close(fig)

    fig, axss = plt.subplots(3, 5, figsize=(5 * 3, 3 * 3), dpi=100)
    for num_particles in num_particless:
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

            # import pdb; pdb.set_trace()
            ax = axss[0, 0]
            plot_errors_end(ax, num_particles, *theta_losses, color=color)
            # ax.plot(moving_average(theta_losses), label=algorithm)
            ax.set_ylabel("model loss")
            ax.set_xticks(num_particless)

            ax = axss[0, 1]
            plot_errors_end(ax, num_particles, *phi_losses, color=color)
            # ax.plot(moving_average(phi_losses), label=algorithm)
            ax.set_ylabel("inference loss")
            ax.set_xticks(num_particless)

            ax = axss[0, 2]
            plot_errors_end(ax, num_particles, *cluster_cov_distances, color=color)
            # ax.plot(cluster_cov_distances, label=algorithm)
            ax.set_ylabel("cluster cov distance")
            ax.set_xticks(num_particless)

            for ax in axss[0, 3:]:
                ax.axis("off")

            ax = axss[1, 0]
            plot_errors_end(ax, num_particles, *test_log_ps, color=color)
            # lines = ax.plot(test_log_ps, label=algorithm)
            ax.axhline(test_log_ps_true[0][0], color="black")
            ax.set_ylabel("TEST\nlog p")
            ax.set_xticks(num_particless)

            ax = axss[1, 1]
            plot_errors_end(ax, num_particles, *test_kl_qps, color=color)
            # ax.plot(test_kl_qps, label=algorithm)
            ax.set_ylabel("KL(q, p)")
            ax.set_xticks(num_particless)

            ax = axss[1, 2]
            plot_errors_end(ax, num_particles, *test_kl_pqs, color=color)
            # ax.plot(test_kl_pqs, label=algorithm)
            ax.set_ylabel("KL(p, q)")
            ax.set_xticks(num_particless)

            ax = axss[1, 3]
            plot_errors_end(ax, num_particles, *test_kl_qps_true, color=color)
            # ax.plot(test_kl_qps_true, label=algorithm)
            ax.set_ylabel("KL(q, p true)")
            ax.set_xticks(num_particless)

            ax = axss[1, 4]
            plot_errors_end(ax, num_particles, *test_kl_pqs_true, color=color)
            # ax.plot(test_kl_pqs_true, label=algorithm)
            ax.set_ylabel("KL(p true, q)")
            ax.set_xticks(num_particless)

            ax = axss[2, 0]
            plot_errors_end(ax, num_particles, *train_log_ps, color=color)
            # lines = ax.plot(train_log_ps, label=algorithm)
            ax.axhline(train_log_ps_true[1][0], color="black")
            ax.set_ylabel("TRAIN\nlog p")
            ax.set_xticks(num_particless)

            ax = axss[2, 1]
            plot_errors_end(ax, num_particles, *train_kl_qps, color=color)
            # lines = ax.plot(train_kl_qps, label=algorithm)
            if algorithm == "mws" or algorithm == "rmws":
                plot_errors_end(
                    ax,
                    num_particles,
                    *train_kl_memory_ps,
                    fillstyle="none",
                    markerfacecolor="white",
                    color=color
                )
                # ax.plot(train_kl_memory_ps, label=algorithm,
                # color=lines[0].get_color(), linestyle='dashed')
            else:
                plot_errors_end(
                    ax,
                    num_particles,
                    *reweighted_train_kl_qps,
                    fillstyle="none",
                    markerfacecolor="white",
                    color=color
                )

            ax.set_ylabel("KL(q, p)")
            ax.set_xticks(num_particless)

            ax = axss[2, 2]
            plot_errors_end(ax, num_particles, *train_kl_pqs, color=color)
            # ax.plot(train_kl_pqs, label=algorithm)
            ax.set_ylabel("KL(p, q)")
            ax.set_xticks(num_particless)

            ax = axss[2, 3]
            plot_errors_end(ax, num_particles, *train_kl_qps_true, color=color)
            # lines = ax.plot(train_kl_qps_true, label=algorithm)
            if algorithm == "mws" or algorithm == "rmws":
                plot_errors_end(
                    ax,
                    num_particles,
                    *train_kl_memory_ps_true,
                    fillstyle="none",
                    markerfacecolor="white",
                    color=color
                )
                # ax.plot(train_kl_memory_ps_true, label=algorithm,
                # color=lines[0].get_color(), linestyle='dashed')
            else:
                plot_errors_end(
                    ax,
                    num_particles,
                    *reweighted_train_kl_qps_true,
                    fillstyle="none",
                    markerfacecolor="white",
                    color=color
                )
            ax.set_ylabel("KL(q, p true)")
            ax.set_xticks(num_particless)

            ax = axss[2, 4]
            plot_errors_end(ax, num_particles, *train_kl_pqs_true, color=color)
            # ax.plot(train_kl_pqs_true, label=algorithm)
            ax.set_ylabel("KL(p true, q)")
            ax.set_xticks(num_particless)

    lines = [
        Line2D([0], [0], color=colors[0], marker="o", linestyle=""),
        Line2D(
            [0],
            [0],
            color=colors[0],
            marker="o",
            linestyle="",
            fillstyle="none",
            markerfacecolor="white",
        ),
        Line2D([0], [0], color=colors[1], marker="o", linestyle=""),
        Line2D([0], [0], color=colors[2], marker="o", linestyle=""),
        # Line2D([0], [0], color=colors[3], marker='o', linestyle=''),
        # Line2D([0], [0], color=colors[3], marker='o', linestyle='', fillstyle='none', markerfacecolor='white'),
    ]
    labels = [
        "MWS",
        "MWS memory",
        "RWS",
        "VIMCO",
        # 'R-MWS',
        # 'R-MWS memory',
    ]
    axss[0, 0].legend(lines, labels)

    for ax in axss[-1]:
        ax.set_xlabel("num particles")
    for axs in axss:
        for ax in axs:
            sns.despine(ax=ax)
            ax.tick_params(direction="in")

    fig.tight_layout(pad=0)
    path = os.path.join(args.diagnostics_dir, "losses.png")
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
