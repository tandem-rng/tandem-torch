# tandem-torch

PyTorch tensors from Tandem8x32. CPU and CUDA fills write the stream of the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for bit, fast on both.

- [API](api.md): the generator, dtypes, normals, exponentials, bounded draws and parallel use.
- [Design](design.md): the CUDA fills and the bounded, normal and exponential contracts.
- [Tests](tests.md): what the suite checks.
- [Speed](speed.md): M4 and A100 figures against torch.

This is not a `torch.Generator`. That class is final and its RNG hooks are internal to
PyTorch, so no third-party generator can drive `torch.rand`. `tandem_torch` fills tensors
through its own functions instead, which is also what makes the stream reproducible across
languages and devices.

## Install

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-torch
pip install torch                      # a CPU or CUDA build, see pytorch.org
pip install --no-build-isolation .
```

The reference sources sit in the `external/tandem-c` (1c75956) and `external/tandem-cuda`
(e98daee) git submodules. Clone with `git clone --recurse-submodules`, or run `git submodule update --init`
in an existing clone. GitHub's ZIP download omits submodules and does not build.

The build needs a C and C++ compiler. With `nvcc` on the path (or `CUDA_HOME` set) the CUDA
fills are built too, otherwise CUDA tensors raise. Set `TANDEM_TORCH_CUDA=0` or `1` to force
the choice. For development, `pixi install` gives a CPU environment, `pixi install -e cuda`
a Linux GPU environment with nvcc 12.8 from conda-forge and PyTorch's cu128 wheel, and
`pixi run test` runs the tests.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
