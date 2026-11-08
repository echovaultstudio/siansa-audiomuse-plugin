"""Checks natten_torch's functions against NATTEN's own (built from source), on random input: 1D and 2D, the
kernel All-In-One uses (5) and others, dilations 1 to 4, lengths from just kernel × dilation up.

    python check_natten.py   # with NATTEN installed, and natten_torch importable as natten_torch.natten
"""

import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("ours", sys.argv[1] if len(sys.argv) > 1 else "natten_torch/natten/functional.py")
ours = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ours)
from natten import functional as _natten  # noqa: E402


class theirs:  # NATTEN's own: its older names when it has them, else its newer na1d/na2d
    @staticmethod
    def natten1dqkrpb(q, k, rpb, kernel, dilation):
        if hasattr(_natten, "natten1dqkrpb"):
            return _natten.natten1dqkrpb(q, k, rpb, kernel, dilation)
        return _natten.na1d_qk(q, k, kernel_size=kernel, dilation=dilation, rpb=rpb)

    @staticmethod
    def natten1dav(p, v, kernel, dilation):
        if hasattr(_natten, "natten1dav"):
            return _natten.natten1dav(p, v, kernel, dilation)
        return _natten.na1d_av(p, v, kernel_size=kernel, dilation=dilation)

    @staticmethod
    def natten2dqkrpb(q, k, rpb, kernel, dilation):
        if hasattr(_natten, "natten2dqkrpb"):
            return _natten.natten2dqkrpb(q, k, rpb, kernel, dilation)
        return _natten.na2d_qk(q, k, kernel_size=kernel, dilation=dilation, rpb=rpb)

    @staticmethod
    def natten2dav(p, v, kernel, dilation):
        if hasattr(_natten, "natten2dav"):
            return _natten.natten2dav(p, v, kernel, dilation)
        return _natten.na2d_av(p, v, kernel_size=kernel, dilation=dilation)


torch.manual_seed(0)
worst = 0.0
cases = 0
for kernel in (3, 5, 7):
    for dilation in (1, 2, 3, 4):
        for extra in (0, 1, 5, 23):
            length = kernel * dilation + extra
            q, k, v = (torch.randn(2, 2, length, 12, dtype=torch.float32) for _ in range(3))
            rpb = torch.randn(2, 2 * kernel - 1, dtype=torch.float32)
            a = ours.natten1dqkrpb(q, k, rpb, kernel, dilation)
            b = theirs.natten1dqkrpb(q, k, rpb, kernel, dilation)
            p = torch.softmax(a, -1)
            worst = max(worst, (a - b).abs().max().item(),
                        (ours.natten1dav(p, v, kernel, dilation) - theirs.natten1dav(p, v, kernel, dilation)).abs().max().item())
            height, width = kernel * dilation + extra % 3, kernel * dilation + extra
            q, k, v = (torch.randn(1, 2, height, width, 12, dtype=torch.float32) for _ in range(3))
            rpb = torch.randn(2, 2 * kernel - 1, 2 * kernel - 1, dtype=torch.float32)
            a = ours.natten2dqkrpb(q, k, rpb, kernel, dilation)
            b = theirs.natten2dqkrpb(q, k, rpb, kernel, dilation)
            p = torch.softmax(a, -1)
            worst = max(worst, (a - b).abs().max().item(),
                        (ours.natten2dav(p, v, kernel, dilation) - theirs.natten2dav(p, v, kernel, dilation)).abs().max().item())
            cases += 2
print(f"{cases} cases, largest difference from NATTEN {worst:.2e}")
sys.exit(0 if worst < 1e-4 else 1)
