<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .pt"></p>

# tandem-torch

[![CI](https://github.com/tandem-rng/tandem-torch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-torch/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

PyTorch tensors from [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
random number generator. CPU and CUDA fills write the specification's stream bit for bit, fast
on both. It is not a `torch.Generator`, which third-party code cannot implement.

The build needs a C and C++ compiler, and `nvcc` or `CUDA_HOME` for the CUDA fills. The
submodules pin tandem-c at 4e9a69f and tandem-cuda at c5c5725.

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-torch
pip install torch                      # a CPU or CUDA build, see pytorch.org
pip install --no-build-isolation .
```

```python
import torch
from tandem_torch import Tandem

t = Tandem(42)                                   # the spec's stream for seed 42
f = t.rand(1 << 20, dtype=torch.float32, device="cuda")
worker = t.split(7)                              # by index, from the key alone
z = worker.randn(1000)                           # Box-Muller, two float64 uniforms each
e = worker.exponential(1000)                     # -log(1 - u), bit exact on CPU and CUDA
```

See [API](docs/api.md) for every draw and dtype, and [tests](docs/tests.md) and
[speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](docs/index.md) · [Apache 2.0 license](LICENSE)
