<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .pt"></p>

# tandem-torch

PyTorch tensors from [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. CPU fills use a
vendored copy of the reference C implementation, CUDA fills a vendored copy of the reference
CUDA header, and both produce the stream the specification defines, bit for bit.

This is not a `torch.Generator`. That class is final and its RNG hooks are internal to
PyTorch, so no third-party generator can drive `torch.rand`. `tandem_torch` fills tensors
through its own functions instead, which is also what makes the stream reproducible across
languages and devices.

## Use

```python
import torch
from tandem_torch import Tandem

t = Tandem(42)                                   # the spec's stream for seed 42     
u = t.rand(1_000_000)                            # float64 in [0, 1), 53 random bits
f = t.rand(1 << 20, dtype=torch.float32, device="cuda")
w = t.bits(1 << 20, dtype=torch.uint32)          # stream words
z = t.randn(1000)                                # inverse CDF of one float64 uniform each
b = t.randbool(100)
t.fill_(torch.empty(4096, dtype=torch.float16))  # fill in place, any supported dtype
worker = t.split(7)                              # by index, from the key alone
kids = t.fork(4)                                 # from the current block, parent moves on
t.key, t.position, t.chunk_length                # transport form
```

Every call aligns the stream position to the element width, reads, and advances, as the
specification requires, so a `uint8` draw followed by a `float64` draw skips to the next
64-bit boundary. The functional forms
`tandem_torch.rand(key, position, *shape, ...)` and `tandem_torch.bits(...)` return
`(tensor, next_position)` for code that keeps the position itself.

Supported dtypes: `bool`, `uint8` to `uint64`, the signed integers as the unsigned draws
reinterpreted, `float16` (`(raw >> 5) * 2^-11`), `float32` (24 random bits) and `float64`
(53 random bits). `randn` is tandem-torch's own convention: `erfinv` of one float64 uniform
shifted by half an ulp into `(0, 1)`, so it consumes 64 stream bits per normal and is finite.

On CUDA, 32-bit and 64-bit types go straight to the tile kernel. Narrower types
and `bool` come from a word fill over the same stream bytes, since the stream is one byte
sequence after alignment.

## Install

```sh
pip install torch   # a CPU or CUDA build, see pytorch.org
pip install --no-build-isolation .
```

The build needs a C and C++ compiler. With `nvcc` on the path (or `CUDA_HOME` set) the CUDA
fills are built too, otherwise CUDA tensors raise. Set `TANDEM_TORCH_CUDA=0` or `1` to force
the choice. For development, `pixi install` gives a CPU environment, `pixi install -e cuda`
a Linux GPU environment with nvcc 12.8 from conda-forge and PyTorch's cu128 wheel, and
`pixi run test` runs the tests.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills from several offsets with reference stream dumps in `tests/data`. With a CUDA device the suite also compares
CUDA fills with CPU fills for every dtype, four chunk lengths, fourteen positions and nine
lengths, and on storage that is not 16-byte aligned. CI runs the CPU tests on Linux and
macOS and fails when the vendored sources or the vectors drift from upstream. The
CUDA tests run by hand on a GPU host.

## Speed

Preallocated outputs, 2^27 elements, `python tools/bench.py [cpu|cuda]`.

Apple M4, one thread, minimum of 7, load 2.2:

| | GiB/s |
|---|---|
| `Tandem.rand` float32 / float64 | 13.2 / 13.1 |
| `torch.rand` float32 / float64 | 3.3 / 5.0 |
| `Tandem.bits` uint32 | 17.1 |
| `torch.randint` int32 | 2.4 |

NVIDIA A100 40 GB PCIe, GPU idle, cudaEvent timings, 0.5 s warm-up, minimum of 21, host load
100 from other users' CPU jobs:

| | GiB/s |
|---|---|
| `Tandem.rand` float32 / float64 | 1300 / 1355 |
| `torch.rand` float32 / float64 | 1101 / 1176 |
| `Tandem.bits` uint32 | 1314 |
| `torch.randint` int32 | 336 |

The CPU path is the C row engine, the CUDA path is the shared-memory tile kernel. At 2^27
elements the tile reads a little below its 2^28 rate because launch and clock ramp are a
larger share of the time.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
