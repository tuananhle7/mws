import util
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def moving_average(x, width=100):
    return np.convolve(x, np.ones(width) / width, 'valid')


def main(args):
    if not os.path.exists(args.diagnostics_dir):
        os.makedirs(args.diagnostics_dir)

    fig, axss = plt.subplots(3, 5, figsize=(5 * 3, 3 * 3), dpi=100)
    for algorithm in ['mws', 'rws']:
        checkpoint_path = '{}_{}.pt'.format(
            args.checkpoint_path_prefix, algorithm)
        (_, _, theta_losses, phi_losses, cluster_cov_distances,
         test_log_ps, test_log_ps_true, test_kl_qps, test_kl_pqs, test_kl_qps_true, test_kl_pqs_true,
         train_log_ps, train_log_ps_true, train_kl_qps, train_kl_pqs, train_kl_qps_true,
         train_kl_pqs_true, train_kl_memory_ps, train_kl_memory_ps_true) = util.load_checkpoint(
            checkpoint_path, torch.device('cpu'))

        ax = axss[0, 0]
        ax.plot(moving_average(theta_losses), label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('model loss')
        ax.set_xticks([0, len(theta_losses) - 1])

        ax = axss[0, 1]
        ax.plot(moving_average(phi_losses), label=algorithm)
        ax.set_xlabel('iteration')
        ax.set_ylabel('inference loss')
        ax.set_xticks([0, len(phi_losses) - 1])

        ax = axss[0, 2]
        ax.plot(cluster_cov_distances, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('cluster cov distance')
        ax.set_xticks([0, len(cluster_cov_distances) - 1])

        for ax in axss[0, 3:]:
            ax.axis('off')

        ax = axss[1, 0]
        lines = ax.plot(test_log_ps, label=algorithm)
        ax.plot(test_log_ps_true, color=lines[0].get_color())
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('TEST\nlog p')
        ax.set_xticks([0, len(test_log_ps) - 1])

        ax = axss[1, 1]
        ax.plot(test_kl_qps, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(q, p)')
        ax.set_xticks([0, len(test_kl_qps) - 1])

        ax = axss[1, 2]
        ax.plot(test_kl_pqs, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(p, q)')
        ax.set_xticks([0, len(test_kl_pqs) - 1])

        ax = axss[1, 3]
        ax.plot(test_kl_qps_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(q, p true)')
        ax.set_xticks([0, len(test_kl_qps_true) - 1])

        ax = axss[1, 4]
        ax.plot(test_kl_pqs_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(p true, q)')
        ax.set_xticks([0, len(test_kl_pqs_true) - 1])

        ax = axss[2, 0]
        lines = ax.plot(train_log_ps, label=algorithm)
        ax.plot(train_log_ps_true, color=lines[0].get_color())
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('TRAIN\nlog p')
        ax.set_xticks([0, len(train_log_ps) - 1])

        ax = axss[2, 1]
        lines = ax.plot(train_kl_qps, label=algorithm)
        if algorithm == 'mws':
            ax.plot(train_kl_memory_ps, label=algorithm,
                    color=lines[0].get_color(), linestyle='dashed')
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(q, p)')
        ax.set_xticks([0, len(train_kl_qps) - 1])

        ax = axss[2, 2]
        ax.plot(train_kl_pqs, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(p, q)')
        ax.set_xticks([0, len(train_kl_pqs) - 1])

        ax = axss[2, 3]
        lines = ax.plot(train_kl_qps_true, label=algorithm)
        if algorithm == 'mws':
            ax.plot(train_kl_memory_ps_true, label=algorithm,
                    color=lines[0].get_color(), linestyle='dashed')
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(q, p true)')
        ax.set_xticks([0, len(train_kl_qps_true) - 1])

        ax = axss[2, 4]
        ax.plot(train_kl_pqs_true, label=algorithm)
        ax.set_xlabel('iteration / 100')
        ax.set_ylabel('KL(p true, q)')
        ax.set_xticks([0, len(train_kl_pqs_true) - 1])

    axss[-1, -1].legend()
    for axs in axss:
        for ax in axs:
            sns.despine(ax=ax, trim=True)
            ax.xaxis.set_label_coords(0.5, -0.01)

    fig.tight_layout(pad=0)
    path = os.path.join(args.diagnostics_dir, 'losses.pdf')
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint-path-prefix', default='checkpoint',
                        help=' ')
    parser.add_argument('--diagnostics-dir', default='diagnostics/', help=' ')
    args = parser.parse_args()
    main(args)
