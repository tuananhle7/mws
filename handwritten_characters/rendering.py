import functools

import torch
import math


def safe_div(x, y, large=1e7):
    return torch.clamp(x / y, -large, large)


def get_center(arcs):
    """Centre of the arc

    Args:
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns: tensor [batch_size, num_arcs, 2]
    """
    x_start = arcs[..., 0]
    y_start = arcs[..., 1]
    dx = arcs[..., 2]
    dy = arcs[..., 3]
    theta = arcs[..., 4]

    x_center = x_start + dx / 2 + safe_div(dy, (2 * torch.tan(theta)))
    y_center = y_start + dy / 2 - safe_div(dx, (2 * torch.tan(theta)))
    return torch.stack([x_center, y_center], dim=-1)


def get_normal(xs):
    """Normals of xs.

    Args:
        xs: tensor [num_xs, 2]

    Returns: tensor [num_xs, 2]
    """

    return torch.stack([xs[:, 1], -xs[:, 0]], dim=-1)


def same_half_plane(normals, x_1, x_2):
    """
    Args:
        normals: tensor [batch_size, num_arcs, 2]
        x_1: tensor [batch_size, num_arcs, 2]
        x_2: tensor [batch_size, num_points, num_arcs, 2]

    Returns: binary tensor [batch_size, num_points, num_arcs]
    """

    return (
        torch.einsum("bai,bai->ba", normals, x_1).unsqueeze(1)
        * torch.einsum("bai,bpai->bpa", normals, x_2)
    ) >= 0


def get_inside_sector(points, arcs):
    """Are `points` inside the sector defined by `arcs`?

    Args:
        points: tensor [num_points, 2]
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns: binary tensor [batch_size, num_points, num_arcs]
    """

    centers = get_center(arcs)  # [batch_size, num_arcs, 2]
    start_points = arcs[..., :2]  # [batch_size, num_arcs, 2]
    end_points = start_points + arcs[..., 2:4]  # [batch_size, num_arcs, 2]

    a = start_points - centers  # [batch_size, num_arcs, 2]
    b = end_points - centers  # [batch_size, num_arcs, 2]
    # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
    # [batch_size, num_points, num_arcs, 2]
    x = points.unsqueeze(1).unsqueeze(0) - centers.unsqueeze(1)

    batch_size, num_arcs, _ = a.shape
    a_normals = get_normal(a.view(batch_size * num_arcs, -1)).view(
        batch_size, num_arcs, -1
    )  # [batch_size, num_arcs, 2]
    b_normals = get_normal(b.view(batch_size * num_arcs, -1)).view(
        batch_size, num_arcs, -1
    )  # [batch_size, num_arcs, 2]

    return same_half_plane(a_normals, b, x) & same_half_plane(b_normals, a, x)


def distance_to_arc(points, arcs):
    """Distance of `points` to the `arcs`.

    Args:
        points: tensor [num_points, 2]
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns:
        distance: tensor [batch_size, num_points, num_arcs]
    """

    centers = get_center(arcs)  # [batch_size, num_arcs, 2] TODO
    start_points = arcs[..., :2]  # [batch_size, num_arcs, 2]
    end_points = start_points + arcs[..., 2:4]  # [batch_size, num_arcs, 2]

    # inside sector
    distance_from_center = torch.norm(
        # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
        # [batch_size, num_points, num_arcs, 2]
        points.unsqueeze(1).unsqueeze(0) - centers.unsqueeze(1),
        p=2,
        dim=-1,
    )  # [batch_size, num_points, num_arcs]
    # [batch_size, num_arcs]
    arc_radius = torch.norm(start_points - centers, p=2, dim=-1)
    # [batch_size, num_points, num_arcs]
    inside_sector_distance = torch.abs(distance_from_center - arc_radius.unsqueeze(1))

    # outside sector
    start_points_distance = torch.norm(
        # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
        # [batch_size, num_points, num_arcs, 2]
        points.unsqueeze(1).unsqueeze(0) - start_points.unsqueeze(1),
        p=2,
        dim=-1,
    )  # [batch_size, num_points, num_arcs]
    end_points_distance = torch.norm(
        # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
        # [batch_size, num_points, num_arcs, 2]
        points.unsqueeze(1).unsqueeze(0) - end_points.unsqueeze(1),
        p=2,
        dim=-1,
    )  # [batch_size, num_points, num_arcs]
    # [batch_size, num_points, num_arcs]
    outside_sector_distance = torch.min(start_points_distance, end_points_distance)

    # [batch_size, num_arcs]
    theta = arcs[:, :, 4]
    # [batch_size, num_points, num_arcs]
    small_arc = (torch.abs(theta[:, None, :]) < (math.pi / 2)).float()
    large_arc = 1 - small_arc

    # [batch_size, num_points, num_arcs]
    inside_sector = get_inside_sector(points, arcs).float()
    outside_sector = 1 - inside_sector
    return small_arc * (
        inside_sector * inside_sector_distance + outside_sector * outside_sector_distance
    ) + large_arc * (
        inside_sector * outside_sector_distance + outside_sector * inside_sector_distance
    )


def get_value(points, arcs, scale, bias):
    """
    Args:
        points: tensor [num_points, 2]
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns:
        value: tensor [batch_size, num_points, num_arcs]
    """
    sharpness = arcs[..., -2]
    width = arcs[..., -1]
    # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L916-L919
    # and
    # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L973-L991
    return (
        scale
        * torch.clamp(
            torch.exp(
                sharpness.unsqueeze(1) ** 2
                * (width.unsqueeze(1) ** 2 - distance_to_arc(points, arcs) ** 2)
            ),
            0,
            1,
        )
        + bias
    )
    # return -sharpness.unsqueeze(1) * (
    #     distance_to_arc(points, arcs) - width.unsqueeze(1)
    # )


@functools.lru_cache(maxsize=None)
def get_grid_points(num_rows, num_cols, dtype, device):
    """Returns grid points.

    Args:
        num_rows: int
        num_cols: int
        dtype
        device

    Returns:
        points: [num_rows * num_cols, 2]
    """
    xs = torch.linspace(0, 1, num_cols, dtype=dtype, device=device)
    ys = torch.linspace(0, 1, num_rows, dtype=dtype, device=device)
    xss, yss = torch.meshgrid(xs, ys)
    points = torch.stack([xss, yss], dim=-1).view(-1, 2)  # [num_points, 2]
    return points


def get_probs(*args, **kwargs):
    return get_logits(*args, **kwargs).sigmoid()


def get_logits(arcs, on_offs, rendering_params, num_rows, num_cols):
    """Turns arc specs into Bernoulli probabilities.

    Args:
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]
        on_offs: long tensor [batch_size, num_arcs]
        rendering_params: tensor [2]
            rendering_params[0] = scale of value
            rendering_params[1] = bias of value
        num_rows: int
        num_cols: int

    Returns:
        probs: tensor [batch_size, num_rows, num_cols]"""

    # [num_points, 2]
    points = get_grid_points(num_rows, num_cols, arcs.dtype, arcs.device)
    batch_size = arcs.shape[0]
    scale, bias = rendering_params
    result = (
        torch.max(
            get_value(points, arcs, scale, bias) + on_offs.float().log().unsqueeze(-2), dim=-1
        )[0]
        .view(batch_size, num_cols, num_rows)
        .transpose(-2, -1)
    )
    epsilon = math.log(1e-6)
    result = torch.logsumexp(
        torch.stack([result, torch.ones(result.shape, device=result.device) * epsilon]), dim=0
    )
    return torch.flip(result, [-2])
