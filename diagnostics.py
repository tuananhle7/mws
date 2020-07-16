import os
import util
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import data
import numpy as np
import models
import math
import random
import rendering
import torch.nn.functional as F
from pathlib import Path

# hack for https://github.com/dmlc/xgboost/issues/1715
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if torch.cuda.is_available():
    device = torch.device("cuda")
    util.logging.info("Using CUDA")
else:
    device = torch.device("cpu")
    util.logging.info("Using CPU")


def plot_losses(
    path,
    theta_losses,
    phi_losses,
    dataset_size,
    condition_on_alphabet,
    prior_losses=None,
    accuracies=None,
    novel_proportions=None,
    new_maps=None,
    batch_split=None,
):
    util.logging.info("plot_losses")
    loss_str = ""
    num_subplots = 2 + len(
        [x for x in [accuracies, novel_proportions, new_maps] if x is not None and len(x) > 0]
    )
    fig, axs = plt.subplots(1, num_subplots, figsize=(2 * num_subplots, 3), dpi=100)
    ax = axs[0]
    ax.plot(theta_losses)
    if prior_losses is not None:
        ax.plot(np.abs(prior_losses))  # abs because I accidentally saved them negative whoops
        if condition_on_alphabet:
            ax.plot(
                np.ones(len(prior_losses)) * math.log(dataset_size / 50),
                color="gray",
                linestyle="dashed",
            )
        else:
            ax.plot(
                np.ones(len(prior_losses)) * math.log(dataset_size),
                color="gray",
                linestyle="dashed",
            )

    ax.set_xlabel("iteration (batch_split={})".format(batch_split))
    ax.set_ylabel("theta loss")
    ax.set_xticks([0, len(theta_losses) - 1])
    if len(theta_losses) > 0:
        loss_str += "theta loss: {:.2f}\n".format(np.mean(theta_losses[:-100]))

    ax = axs[1]
    ax.plot(phi_losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("phi loss")
    ax.set_xticks([0, len(phi_losses) - 1])
    if len(phi_losses) > 0:
        loss_str += "phi loss: {:.2f}\n".format(np.mean(phi_losses[:-100]))

    plot_idx = 2
    if accuracies is not None and len(accuracies) > 0:
        ax = axs[plot_idx]
        ax.plot(accuracies)
        ax.set_xlabel("iteration")
        ax.set_ylabel("accuracy")
        ax.set_xticks([0, len(accuracies) - 1])
        if len(accuracies) > 0:
            loss_str += "accuracy: {:.2f}%\n".format(np.mean(accuracies[:-100]) * 100)
        plot_idx = plot_idx + 1

    if novel_proportions is not None and len(novel_proportions) > 0:
        ax = axs[plot_idx]
        ax.plot(novel_proportions)
        ax.set_xlabel("iteration")
        ax.set_ylabel("novel_proportion")
        ax.set_xticks([0, len(novel_proportions) - 1])
        if len(novel_proportions) > 0:
            loss_str += "novel_proportion: {:.2f}%\n".format(
                np.mean(novel_proportions[:-100]) * 100
            )
        plot_idx = plot_idx + 1

    if new_maps is not None and len(new_maps) > 0:
        ax = axs[plot_idx]
        ax.plot(new_maps)
        ax.set_xlabel("iteration")
        ax.set_ylabel("new MAP?")
        ax.set_ylim(0, 0.2)
        ax.set_xticks([0, len(new_maps) - 1])
        if len(new_maps) > 0:
            loss_str += "new_map: {:.2f}%\n".format(np.mean(new_maps[:-100]) * 100)
        plot_idx = plot_idx + 1

    loss_str += "iteration: {}".format(len(theta_losses))

    with open(path + ".txt", "w") as text_file:
        text_file.write(loss_str)

    for ax in axs:
        sns.despine(ax=ax, trim=True)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def plot_log_ps(path, log_ps):
    util.logging.info("plot_log_ps")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    ax.plot(log_ps)
    ax.set_xlabel("iteration")
    ax.set_ylabel("log p")
    sns.despine(ax=ax, trim=True)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def plot_kls(path, kls):
    util.logging.info("plot_kls")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    ax.plot(kls)
    ax.set_xlabel("iteration")
    ax.set_ylabel("KL")
    sns.despine(ax=ax, trim=True)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def plot_reconstructions(
    path,
    generative_model,
    inference_network,
    num_reconstructions,
    dataset,
    data_location,
    dataset_size,
    memory=None,
    legacy_index=False,
    resolution=28,
    obs_ids=None,
):
    show_obs_ids = obs_ids is None
    util.logging.info("plot_reconstructions")
    if dataset == "mnist":
        data_train, _, _ = data.load_binarized_mnist(location=data_location)
    elif dataset == "omniglot":
        (
            data_train,
            data_valid,
            data_test,
            target_train,
            target_valid,
            target_test,
        ) = data.load_binarized_omniglot_with_targets(location=args.data_location)
    if dataset_size is not None:
        if dataset == "mnist":
            data_train = data_train[:dataset_size]
        elif dataset == "omniglot":
            if legacy_index:
                data_train = data_train[:dataset_size]
            else:
                data_train, target_train = data.split_data_by_target(
                    data_train, target_train, num_data_per_target=dataset_size // 50
                )
    if memory is None:
        ids = False
    else:
        ids = True
    data_train = torch.tensor(data_train, device=device)
    target_train = torch.tensor(target_train, device=device)
    if obs_ids is None:
        obs_id = torch.tensor(
            np.random.choice(np.arange(len(data_train)), num_reconstructions), device=device
        ).long()
    else:
        obs_id = torch.tensor(obs_ids, device=device).long()
        num_reconstructions = len(obs_ids)

    obs = data_train[obs_id].float().view(-1, 28, 28)
    if generative_model.use_alphabet:
        alphabet = torch.tensor(target_train[obs_id], device=device).float()
        obs = (obs, alphabet)

    start_point = torch.Tensor([0.5, 0.5]).unsqueeze(0).expand(num_reconstructions, -1).to(device)
    if memory is None:
        # [batch_size, num_arcs, 2]
        latent = inference_network.sample(obs, 1)[0]
        latent2 = inference_network.sample(obs, 1)[0]
    else:
        # [batch_size, memory_size, num_arcs, 2]
        memory_latent = memory[obs_id]
        memory_latent_transposed = memory_latent.transpose(
            0, 1
        ).contiguous()  # [memory_size, batch_size, num_arcs, 2]
        memory_log_p = generative_model.get_log_prob(
            memory_latent_transposed, obs, obs_id
        )  # [memory_size, batch_size]
        dist = torch.distributions.Categorical(
            probs=util.exponentiate_and_normalize(memory_log_p.t(), dim=1)
        )
        sampled_memory_id = dist.sample()  # [batch_size]
        latent = torch.gather(
            memory_latent_transposed,
            0,
            sampled_memory_id[None, :, None, None].repeat(1, 1, generative_model.num_arcs, 2),
        )[
            0
        ]  # [batch_size, num_arcs, 2]
        # sampled_memory_id2 = dist.sample()  # [batch_size]
        # latent2 = torch.gather(
        #    memory_latent_transposed,
        #    0,
        #    sampled_memory_id2[None, :, None, None].repeat(
        #        1, 1, generative_model.num_arcs, 2)
        # )[0]  # [batch_size, num_arcs, 2]

    # log_prob, accuracy = generative_model.get_log_prob(
    #    latent2[None], obs, obs_id, get_accuracy=True)
    # sort_by = accuracy*100 if accuracy is not None else log_prob[0]
    # sort_idxs = torch.sort(sort_by, descending=True).indices
    # obs = obs[sort_idxs]
    # obs_id = obs_id[sort_idxs]
    # latent = latent[sort_idxs]
    # latent2 = latent2[sort_idxs]
    # sort_by = sort_by[sort_idxs]

    ids = latent[..., 0]
    on_offs = latent[..., 1]
    num_arcs = ids.shape[1]

    reconstructed_images = []
    for arc_id in range(num_arcs):
        reconstructed_images.append(
            models.get_image_probs(
                ids[:, : (arc_id + 1)],
                on_offs[:, : (arc_id + 1)],
                start_point,
                generative_model.get_primitives(),
                generative_model.get_rendering_params(),
                resolution,
                resolution,
            ).detach()
        )

    ### plot another sample
    # ids2 = latent2[..., 0]
    # on_offs2 = latent2[..., 1]
    # reconstructed_images.append(
    #    models.get_image_probs(
    #        ids2[:, :(num_arcs)], on_offs2[:, :(num_arcs)],
    #        start_point, generative_model.get_primitives(),
    #        generative_model.get_rendering_params(),
    #        resolution, resolution
    #    ).detach()
    # )

    fig, axss = plt.subplots(
        num_reconstructions,
        num_arcs + 1,
        figsize=((num_arcs + 1) * 2, num_reconstructions * 2),
        sharex=False,
        sharey=False,
    )
    for i, axs in enumerate(axss):
        if generative_model.use_alphabet:
            obs_ = obs[0][i]
        else:
            obs_ = obs[i]
        axs[0].imshow(obs_.cpu(), "Greys", vmin=0, vmax=1)
        for j, ax in enumerate(axs[1:]):
            l = len(axs[1:])
            char_on = on_offs[i].cpu()
            for k in range(num_arcs):
                if char_on[-k]:
                    l = l - 1
                if l == j:
                    ax.imshow(reconstructed_images[-k][i].cpu(), "Greys", vmin=0, vmax=1)
                    break

        if show_obs_ids:
            axs[0].text(
                0.5,
                0.99,
                str(obs_id[i].item()),
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=12,
            )
            #    ax.text(
            #        .5, .99,
            #        "({:.2f})".format(sort_by[i].item()),
            #        horizontalalignment='center', verticalalignment='top',
            #        transform=ax.transAxes, fontsize=12)

            # if j < num_arcs:
            #    ax.text(
            #        .99, .99,
            #        '{} {}'.format(ids[i, j], 'ON' if on_offs[i, j] else 'OFF'),
            #        horizontalalignment='right', verticalalignment='top',
            #        transform=ax.transAxes)
            # else:
            #    ax.text(
            #        .5, .99,
            #        "({:.2f})".format(sort_by[i].item()),
            #        horizontalalignment='center', verticalalignment='top',
            #        transform=ax.transAxes, fontsize=12)

    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)

    axss[0, 0].set_title(r"Image $x$")
    axss[0, 1].set_title("Sequential\nreconstruction ->")
    fig.tight_layout(pad=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def plot_primitives(path, generative_model):
    util.logging.info("plot_primitives")
    num_primitives = generative_model.num_primitives
    ids = torch.arange(num_primitives, device=device).long().unsqueeze(1)
    on_offs = torch.ones(num_primitives, device=device).long().unsqueeze(1)
    start_point = torch.Tensor([0.5, 0.5]).unsqueeze(0).repeat(num_primitives, 1).to(device)

    primitives = generative_model.get_primitives()
    rendering_params = generative_model.get_rendering_params()
    primitives_imgs = models.get_image_probs(
        ids,
        on_offs,
        start_point,
        primitives,
        rendering_params,
        generative_model.num_rows,
        generative_model.num_cols,
    ).detach()

    num_rows = math.floor(math.sqrt(num_primitives))
    num_cols = num_rows
    fig, axss = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2), sharex=True, sharey=True
    )

    for i, axs in enumerate(axss):
        for j, ax in enumerate(axs):
            ax.imshow(primitives_imgs[i * num_cols + j].cpu(), "Greys", vmin=0, vmax=1)
            ax.text(
                0.99,
                0.99,
                "{}".format(i * num_cols + j),
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(pad=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def plot_prior(path, generative_model, num_samples, resolution=28):
    util.logging.info("plot_prior")
    try:
        latent, _ = generative_model.sample_latent_and_obs(num_samples=num_samples)
    except NotImplementedError:
        util.logging.info("Can't plot samples for this model")
        return
    generative_model.num_rows, generative_model.num_cols = resolution, resolution
    obs_probs = generative_model.get_obs_params(latent).sigmoid().detach()
    generative_model.num_rows, generative_model.num_cols = 28, 28

    num_rows = math.floor(math.sqrt(num_samples))
    num_cols = num_rows
    if num_rows * num_cols < num_samples:
        util.logging.info(
            "Plotting {} * {} samples instead of {} samples.".format(
                num_rows, num_cols, num_samples
            )
        )
    fig, axss = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
    for i, axs in enumerate(axss):
        for j, ax in enumerate(axs):
            ax.imshow(obs_probs[i * num_cols + j].cpu(), cmap="Greys", vmin=0, vmax=1)
            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)
            ax.set_xticks([])
            ax.set_yticks([])
            arc_ids = latent[i * num_cols + j][:, 0]
            on_off_ids = latent[i * num_cols + j][:, 1]
            latent_str = " ".join(
                [
                    "{}{}".format(int(arc_id.item()), "" if int(on_off_id.item()) == 1 else "-")
                    for arc_id, on_off_id in zip(arc_ids, on_off_ids)
                ]
            )
            # ax.text(
            #    .99, .99,
            #    latent_str,
            #    horizontalalignment='right', verticalalignment='top',
            #    transform=ax.transAxes,
            #    fontsize=6)

    fig.tight_layout(pad=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def plot_renderer(path):
    util.logging.info("plot_renderer")
    num_interpolations = 20

    dx = torch.linspace(-0.4, 0.4, num_interpolations)
    dy = torch.linspace(-0.4, 0.4, num_interpolations)
    theta = torch.linspace(-0.99 * math.pi, 0.99 * math.pi, num_interpolations)
    sharpness = torch.linspace(-20, 20, num_interpolations)
    width = torch.linspace(0, 0.1, num_interpolations)
    scale = torch.linspace(0, 10, num_interpolations)
    bias = torch.linspace(-10, 10, num_interpolations)
    interpolations = [dx, dy, theta, sharpness, width]
    interpolations_str = ["dx", "dy", "theta", "sharpness", "width"]

    dx_constant = 0.2
    dy_constant = 0.2
    theta_constant = math.pi / 3
    sharpness_constant = 20.0
    width_constant = 0.01
    scale_constant = 3.0
    bias_constant = -3
    constants = [dx_constant, dy_constant, theta_constant, sharpness_constant, width_constant]

    num_rows, num_cols = 100, 100
    fig, axss = plt.subplots(
        7, num_interpolations, figsize=(num_interpolations * 2, 7 * 2), dpi=200
    )

    for interpolation_id in range(2, 3):
        # [num_interpolations, 1, 7]
        arcs = torch.cat(
            [torch.tensor(0.5).repeat(num_interpolations, 2)]
            + [
                torch.tensor(constants[i]).repeat(num_interpolations, 1)
                for i in range(interpolation_id)
            ]
            + [interpolations[interpolation_id][:, None]]
            + [
                torch.tensor(constants[i]).repeat(num_interpolations, 1)
                for i in range(interpolation_id + 1, 5)
            ],
            dim=1,
        )[:, None, :]

        # [num_interpolations, 1]
        on_offs = torch.ones(num_interpolations, 1).long()

        # [2]
        rendering_params = torch.tensor([scale_constant, bias_constant])

        probs = rendering.get_probs(arcs, on_offs, rendering_params, num_rows, num_cols)
        for i in range(num_interpolations):
            ax = axss[interpolation_id, i]
            ax.imshow(probs[i], vmin=0, vmax=1, cmap="Greys")
            if interpolations_str[interpolation_id] == "theta":
                interpolation = "{:.2f}°".format(
                    math.degrees(interpolations[interpolation_id][i].item())
                )
            else:
                interpolation = "{:.2f}".format(interpolations[interpolation_id][i])
            # ax.text(
            #    .99, .99,
            #    '{} = {}'.format(
            #        interpolations_str[interpolation_id], interpolation),
            #    horizontalalignment='right', verticalalignment='top',
            #    transform=ax.transAxes)

            # Rendering params
            # [1, 1, 7]
            arcs = torch.cat(
                [torch.tensor(0.5).repeat(1, 2)]
                + [torch.tensor(constants[i]).repeat(1, 1) for i in range(5)],
                dim=1,
            )[:, None, :]

            # [1, 1]
            on_offs = torch.ones(1, 1).long()

            # Scale
            # [2]
            # rendering_params = torch.tensor([scale[i], bias_constant])
            # prob = rendering.get_probs(
            #    arcs, on_offs, rendering_params, num_rows, num_cols)[0]
            # ax = axss[5, i]
            # ax.imshow(prob, vmin=0, vmax=1, cmap='Greys')
            # ax.text(
            #    .99, .99,
            #    'scale = {:.2f}'.format(scale[i]),
            #    horizontalalignment='right', verticalalignment='top',
            #    transform=ax.transAxes)

            ## Bias
            ## [2]
            # rendering_params = torch.tensor([scale_constant, bias[i]])
            # prob = rendering.get_probs(
            #    arcs, on_offs, rendering_params, num_rows, num_cols)[0]
            # ax = axss[6, i]
            # ax.imshow(prob, vmin=0, vmax=1, cmap='Greys')
            # ax.text(
            #    .99, .99,
            #    'bias = {:.2f}'.format(bias[i]),
            #    color='grey',
            #    horizontalalignment='right', verticalalignment='top',
            #    transform=ax.transAxes)

    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


# helper function for plot_alphabets
def tile(obs, num_rows, num_cols, resolution=28):
    result = torch.zeros(num_rows * resolution, num_cols * resolution)
    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < len(obs):
                result[
                    i * resolution : (i + 1) * resolution, j * resolution : (j + 1) * resolution
                ] = obs[i * num_cols + j]
    return result


# helper function for plot_alphabets
def get_obs_probs_from_alphabet(generative_model, alphabet_id, num_samples, resolution=28):
    alphabet = F.one_hot(alphabet_id, 50).float()
    latent, obs = generative_model.sample_latent_and_obs(alphabet[None], num_samples)
    latent = latent[:, 0, :, :]
    obs = obs[:, 0, :, :]
    generative_model.num_rows, generative_model.num_cols = resolution, resolution
    obs_probs = generative_model.get_obs_params(latent).sigmoid().detach()
    generative_model.num_rows, generative_model.num_cols = 28, 28
    return obs_probs


# helper function for plot_alphabets
def get_reconstructions(
    obs_id, obs, generative_model, inference_network, memory=None, resolution=28
):
    batch_size = len(obs_id)
    start_point = torch.tensor([0.5, 0.5], device=device).unsqueeze(0).expand(batch_size, -1)
    if memory is None:
        # [batch_size, num_arcs, 2]
        latent = inference_network.sample(obs, 1)[0]
    else:
        # [batch_size, memory_size, num_arcs, 2]
        memory_latent = memory[obs_id]
        memory_latent_transposed = memory_latent.transpose(
            0, 1
        ).contiguous()  # [memory_size, batch_size, num_arcs, 2]
        memory_log_p = generative_model.get_log_prob(
            memory_latent_transposed, obs, obs_id
        )  # [memory_size, batch_size]
        dist = torch.distributions.Categorical(
            probs=util.exponentiate_and_normalize(memory_log_p.t(), dim=1)
        )
        sampled_memory_id = dist.sample()  # [batch_size]
        latent = torch.gather(
            memory_latent_transposed,
            0,
            sampled_memory_id[None, :, None, None].repeat(1, 1, generative_model.num_arcs, 2),
        )[
            0
        ]  # [batch_size, num_arcs, 2]

    ids = latent[..., 0]
    on_offs = latent[..., 1]

    return models.get_image_probs(
        ids,
        on_offs,
        start_point,
        generative_model.get_primitives(),
        generative_model.get_rendering_params(),
        resolution,
        resolution,
    ).detach()


def plot_alphabets(
    path,
    generative_model,
    dataset_size,
    data_location,
    inference_network,
    memory=None,
    legacy_index=False,
    resolution=28,
    alphabet_ids=None,
    num_rows=4,
    num_cols=5,
):

    show_alphabet_ids = alphabet_ids is None
    util.logging.info("plot_alphabets")
    (
        data_train,
        data_valid,
        data_test,
        target_train,
        target_valid,
        target_test,
    ) = data.load_binarized_omniglot_with_targets(location=data_location)

    if dataset_size is not None:
        if legacy_index:
            data_train = data_train[:dataset_size]
            target_train = target_train[:dataset_size]
        else:
            data_train, target_train = data.split_data_by_target(
                data_train, target_train, num_data_per_target=dataset_size // 50
            )

    data_train = torch.tensor(data_train, dtype=torch.float, device=device)
    target_train = torch.tensor(target_train, dtype=torch.float, device=device)
    target_train_numeric = torch.mv(target_train, torch.arange(50, device=device).float()).long()

    if alphabet_ids is None:
        num_alphabets = 50
        alphabet_ids = torch.arange(num_alphabets, device=device)
    else:
        num_alphabets = len(alphabet_ids)
        alphabet_ids = torch.tensor(alphabet_ids, device=device).long()
    # alphabet_ids = torch.randint(50, (num_alphabets,))

    num_obs_per_alphabet = num_rows * num_cols
    num_generalizations = 1
    fig, axss = plt.subplots(
        num_alphabets,
        (2 + num_generalizations),
        figsize=((2 + num_generalizations) * num_cols / 2, num_alphabets * num_rows / 2),
        dpi=200,
    )

    for axs, alphabet_id in zip(axss, alphabet_ids):
        axs[0].imshow(
            tile(
                data_train[target_train_numeric == alphabet_id][:num_obs_per_alphabet],
                num_rows,
                num_cols,
                28,
            ),
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        torch.manual_seed(0)
        axs[1].imshow(
            tile(
                get_reconstructions(
                    obs_id=torch.arange(len(data_train), device=device)[target_train_numeric == alphabet_id],
                    obs=(
                        data_train[target_train_numeric == alphabet_id],
                        target_train[target_train_numeric == alphabet_id],
                    ),
                    generative_model=generative_model,
                    inference_network=inference_network,
                    memory=memory,
                    resolution=resolution,
                ),
                num_rows,
                num_cols,
                resolution,
            ),
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        for i in range(2, num_generalizations + 2):
            try:
                axs[i].imshow(
                    tile(
                        get_obs_probs_from_alphabet(
                            generative_model, alphabet_id, num_obs_per_alphabet, resolution
                        ),
                        num_rows,
                        num_cols,
                        resolution=resolution,
                    ),
                    cmap="Greys",
                    vmin=0,
                    vmax=1,
                )
            except NotImplementedError:
                util.logging.info("Can't sample from this model")
                return
        if show_alphabet_ids:
            axs[0].text(
                0.5,
                0.99,
                str(alphabet_id.item()),
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=axs[0].transAxes,
                fontsize=10,
            )

    for axs in axss:
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)
    axss[0, 0].set_title("Alphabet")
    axss[0, 1].set_title("Reconstructions")
    axss[0, 2].set_title("Generalizations")

    fig.tight_layout(pad=1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def main(args):
    # plot_renderer('{}/renderer.pdf'.format(args.diagnostics_dir))
    dataset = "omniglot"

    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths())
    else:
        checkpoint_paths = [args.checkpoint_path]

    if args.shuffle:
        random.shuffle(checkpoint_paths)

    for checkpoint_path in checkpoint_paths:
        try:
            (
                generative_model,
                inference_network,
                optimizer,
                memory,
                stats,
                run_args,
            ) = util.load_checkpoint(checkpoint_path, device=device)
        except FileNotFoundError as e:
            print(e)
            if "No such file or directory" in str(e):
                print(e)
                continue

        iteration = len(stats.theta_losses)
        if run_args.small_dataset:
            dataset_size = 500
        else:
            if run_args.dataset_size is None:
                (
                    data_train,
                    data_valid,
                    data_test,
                    target_train,
                    target_valid,
                    target_test,
                ) = data.load_binarized_omniglot_with_targets(location=args.data_location)
                dataset_size = data_train.shape[0]
            else:
                dataset_size = run_args.dataset_size
        legacy_index = False

        diagnostics_dir = util.get_save_dir(run_args)
        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

        if run_args.condition_on_alphabet and dataset == "omniglot":
            plot_alphabets(
                f"{diagnostics_dir}/alphabets/{iteration}.pdf",
                generative_model,
                dataset_size,
                run_args.data_location,
                inference_network,
                memory,
                legacy_index,
                resolution=args.resolution,
                num_rows=args.alphabet_num_rows,
                num_cols=args.alphabet_num_cols,
            )

        plot_losses(
            f"{diagnostics_dir}/losses.pdf",
            stats.theta_losses,
            stats.phi_losses,
            dataset_size,
            run_args.condition_on_alphabet,
            stats.prior_losses,
            stats.accuracies,
            stats.novel_proportions,
            stats.new_maps,
            1,
        )
        if args.loss_only:
            continue
        if len(stats.kls) > 0:
            plot_log_ps(f"{diagnostics_dir}/logp.pdf", stats.log_ps)
            plot_kls(f"{diagnostics_dir}/kl.pdf", stats.kls)

        if not run_args.condition_on_alphabet:
            plot_reconstructions(
                f"{diagnostics_dir}/reconstructions/{iteration}.pdf",
                generative_model,
                inference_network,
                args.num_reconstructions,
                dataset,
                args.data_location,
                dataset_size,
                memory,
                legacy_index,
                args.resolution,
            )

        plot_primitives(f"{diagnostics_dir}/primitives/{iteration}.pdf", generative_model)
        if (
            (not generative_model.pixelcnn_likelihood)
            and (not run_args.condition_on_alphabet)
            and (not generative_model.likelihood == "classify")
        ):
            plot_prior(
                f"{diagnostics_dir}/prior/{iteration}.pdf",
                generative_model,
                args.num_prior_samples,
            )
        print("-----------------")


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--repeat", action="store_true", help="")
    parser.add_argument("--shuffle", action="store_true", help="")

    # load checkpoint
    parser.add_argument("--checkpoint-path", type=str, default=None, help=" ")
    parser.add_argument("--data-location", default="local", help=" ")

    # plot
    parser.add_argument("--diagnostics-dir", default="diagnostics", help=" ")
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=28)
    parser.add_argument("--num-reconstructions", type=int, default=20, help=" ")
    parser.add_argument("--num-prior-samples", type=int, default=100)
    parser.add_argument("--alphabet-num-rows", type=int, default=4)
    parser.add_argument("--alphabet-num-cols", type=int, default=5)

    # for custom checkpoints
    parser.add_argument("--checkpoint-iteration", default=None, type=int)
    parser.add_argument("--dataset-size", default=500, type=int)
    parser.add_argument("--alphabet-ids", default=None, type=str)
    parser.add_argument("--obs-ids", default=None, type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with torch.no_grad():
        if args.repeat:
            while True:
                main(args)
        else:
            main(args)
