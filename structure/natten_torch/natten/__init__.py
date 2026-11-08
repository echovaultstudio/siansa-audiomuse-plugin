"""The four neighbourhood-attention functions All-In-One uses, in plain PyTorch, installed as ``natten``.

NATTEN (github.com/SHI-Labs/NATTEN) builds them as CUDA and C++ kernels, published as wheels for each torch and
CUDA on its own site, whose certificate has expired; built from source they take long and need CUDA's compilers.
All-In-One calls only natten1dqkrpb / natten1dav / natten2dqkrpb / natten2dav (NATTEN 0.14–0.17's functions), with
odd kernels, and pads its input to at least kernel × dilation first. These do the same in PyTorch, on the CPU or the
GPU: slower than NATTEN's kernels, which matters little next to separating the song. They're checked against
NATTEN itself, built from source, by ../check_natten.py in CI.
"""

__version__ = "0.17.1+torch"
