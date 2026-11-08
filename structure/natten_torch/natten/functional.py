"""natten1dqkrpb, natten1dav, natten2dqkrpb, natten2dav, as NATTEN 0.14–0.17 define them.

Each position attends to the ``kernel_size`` positions around it among those ``dilation`` apart from it (its
dilation group); at a group's ends the window is moved inwards so it keeps ``kernel_size`` positions. The relative
positional bias ``rpb`` (2·kernel_size − 1 per axis) is indexed by where the neighbour is from the position, so a
window moved at an end uses the bias of where its neighbours really are.
"""

import torch


def _window(length, kernel_size, dilation, device):
    """``(neighbours, bias)``: for each of ``length`` positions, the ``kernel_size`` positions it attends to, and
    the index of each one's relative positional bias (0 .. 2·kernel_size − 2)."""
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size {kernel_size} must be odd")
    if length < kernel_size * dilation:
        raise ValueError(f"length {length} is less than kernel_size × dilation ({kernel_size} × {dilation})")
    position = torch.arange(length, device=device)
    group = position % dilation
    index = position // dilation  # the position within its dilation group
    group_length = (length - group + dilation - 1) // dilation
    start = torch.minimum(torch.clamp(index - kernel_size // 2, min=0), group_length - kernel_size)
    within = start[:, None] + torch.arange(kernel_size, device=device)[None, :]
    neighbours = group[:, None] + dilation * within
    bias = within - index[:, None] + (kernel_size - 1)
    return neighbours, bias


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """query, key: (batch, heads, length, dim); rpb: (heads, 2k − 1) or None → (batch, heads, length, k)."""
    neighbours, bias = _window(query.shape[2], kernel_size, dilation, query.device)
    scores = torch.einsum("bhld,bhlkd->bhlk", query, key[:, :, neighbours, :])
    if rpb is not None:
        scores = scores + rpb[:, bias].unsqueeze(0).to(scores.dtype)
    return scores


def natten1dav(attention, value, kernel_size, dilation):
    """attention: (batch, heads, length, k); value: (batch, heads, length, dim) → (batch, heads, length, dim)."""
    neighbours, _ = _window(value.shape[2], kernel_size, dilation, value.device)
    return torch.einsum("bhlk,bhlkd->bhld", attention, value[:, :, neighbours, :])


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """query, key: (batch, heads, height, width, dim); rpb: (heads, 2k − 1, 2k − 1) or None →
    (batch, heads, height, width, k·k), the neighbours row by row."""
    rows, row_bias = _window(query.shape[2], kernel_size, dilation, query.device)
    cols, col_bias = _window(query.shape[3], kernel_size, dilation, query.device)
    near = key[:, :, rows[:, None, :, None], cols[None, :, None, :], :]  # (b, h, H, W, k, k, d)
    scores = torch.einsum("bhxyd,bhxyijd->bhxyij", query, near)
    if rpb is not None:
        scores = scores + rpb[:, row_bias[:, None, :, None], col_bias[None, :, None, :]].unsqueeze(0).to(scores.dtype)
    return scores.flatten(-2)


def natten2dav(attention, value, kernel_size, dilation):
    """attention: (batch, heads, height, width, k·k); value: (batch, heads, height, width, dim) →
    (batch, heads, height, width, dim)."""
    rows, _ = _window(value.shape[2], kernel_size, dilation, value.device)
    cols, _ = _window(value.shape[3], kernel_size, dilation, value.device)
    near = value[:, :, rows[:, None, :, None], cols[None, :, None, :], :]
    weights = attention.unflatten(-1, (kernel_size, kernel_size))
    return torch.einsum("bhxyij,bhxyijd->bhxyd", weights, near)
