<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .pt"></p>

# tandem-torch

PyTorch tensors from [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. CPU fills use the
reference C implementation, CUDA fills the reference CUDA header, and both produce the stream
the specification defines, bit for bit.

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
z = t.randn(1000)                                # Box-Muller, two float64 uniforms each
b = t.randbool(100)
i = t.randint(-5, 5, (4, 4))                     # int64 on [-5, 5), Lemire, same values on CUDA
p = t.randperm(10)                               # Fisher-Yates, defined on the CPU
x = t.shuffle(torch.arange(12).view(3, 4), dim=1)
t.fill_(torch.empty(4096, dtype=torch.float16))  # fill in place, any supported dtype
x = t.at(torch.float32, 1000)                    # element 1000 of the next float32 fill, no advance
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
(53 random bits), and `complex64` and `complex128`, whose element is the real then the
imaginary component as in the specification. `bfloat16` is a tandem-torch extension that is not in
the specification: `(raw16 >> 8) * 2^-8` of a 16-bit word, the float16 rule with 8 fraction
bits.

`randn` is Box-Muller as in tandem-cuda, not part of the specification. Elements 2j and 2j + 1
are the cos and sin halves of one step from the uniforms 2j and 2j + 1 of the plain float fill,
so an odd count still consumes both uniforms of its last pair. A float64 pair takes 128 stream
bits and a float32 pair 64, with float32 computed in float (`tandem_fill_normal_f64` and `_f32`
on CPU, `tandem::fill_normal_f64` and `_f32` on CUDA). Library `sin`, `cos` and `log` differ in
the last bits, so devices agree to a few ulps and not bit for bit. `float16` and `bfloat16`
round the float32 normal. Empty bounded and normal fills leave the position alone.

`randint(low, high, size, dtype=torch.int64, device="cpu")` draws on `[low, high)`. A range of
at most 2^32 takes one 32-bit draw per element (`tandem_fill_u32_below` on CPU,
`tandem::fill_u32_below` on CUDA), a larger range one 64-bit draw. Both use Lemire's method and
a rejected draw retries on a fallback stream, so CPU and CUDA give the same values and every
element consumes exactly one draw. `randperm(n)` is Fisher-Yates from the end with one
sequential scalar bounded draw per swap, `tandem_u32_below` over the stream, so the CPU
defines it and a CUDA result is copied from there. `shuffle(x, dim=0)` indexes `x` with a
`randperm`. None of these is in the specification, and `randperm` needs `n < 2^32`.

`at(dtype, i)` is random access: element `i` of the fill of `dtype` that would start at the
current position, as a Python number, computed on the host without advancing. It supports
`int32`, `uint32`, `int64`, `uint64`, `float32` and `float64`, the types random access has in
tandem-c and tandem-cuda.

On CUDA every dtype has its own fill kernel in `tandem.cuh`, including `bool`, the 8-bit and
16-bit types and `float16`. `bfloat16` is the 16-bit word fill followed by the scaling.

## Install

```sh
pip install torch   # a CPU or CUDA build, see pytorch.org
pip install --no-build-isolation .
```

The reference sources sit in the `external/tandem-c` and `external/tandem-cuda` git
submodules. Clone with `git clone --recurse-submodules`, or run `git submodule update --init`
in an existing clone. GitHub's ZIP download omits submodules and does not build.

The build needs a C and C++ compiler. With `nvcc` on the path (or `CUDA_HOME` set) the CUDA
fills are built too, otherwise CUDA tensors raise. Set `TANDEM_TORCH_CUDA=0` or `1` to force
the choice. For development, `pixi install` gives a CPU environment, `pixi install -e cuda`
a Linux GPU environment with nvcc 12.8 from conda-forge and PyTorch's cu128 wheel, and
`pixi run test` runs the tests.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills from several offsets with the reference
stream dumps in `tests/data`, complex types included. It checks `randint` and `randn` against
`tests/cross.json`, which `tools/cross_json.py` makes from the cross-check headers of the
submodules and which holds the values of tandem-c and tandem-cuda, `randperm` against a
Python Fisher-Yates over the stream words, `bfloat16` against the 16-bit word fill and `at`
against fills. With a CUDA device the suite also runs the cross-checks there and compares CUDA
fills with CPU fills for every dtype, four chunk lengths, fourteen positions and nine lengths,
and on storage that is not 16-byte aligned. CI runs the CPU tests on Linux and macOS and fails
when the vectors drift from upstream or a submodule pin is not on its upstream main.
`tools/bump.sh` moves the pins to the latest main. The CUDA tests run by hand on a GPU host.

## Speed

Preallocated outputs, 2^27 elements, `python tools/bench.py [cpu|cuda]`.

Apple M4, one thread, minimum of 7:

| | GiB/s |
|---|---|
| `Tandem.rand` float32 / float64 | 17.4 / 17.6 |
| `torch.rand` float32 / float64 | 3.3 / 4.8 |
| `Tandem.bits` uint32 | 20.2 |
| `torch.randint` int32 | 2.3 |

NVIDIA A100 40 GB PCIe, GPU idle, cudaEvent timings, 0.5 s warm-up, minimum of 21:

| | GiB/s |
|---|---|
| `Tandem.rand` float32 / float64 | 1330 / 1364 |
| `torch.rand` float32 / float64 | 1100 / 1197 |
| `Tandem.bits` uint32 | 1334 |
| `Tandem` fill uint8 / bool / float16 | 1174 / 1100 / 1313 |
| `torch.randint` int32 | 332 |
| `Tandem.randint` int32 | 393 |
| `Tandem.randn` float32 / float64 | 1117 / 694 |
| `torch.randn` float32 / float64 | 767 / 569 |

`Tandem.randint` and `Tandem.randn` allocate their result, the `torch` calls and the other
rows write into a preallocated tensor.

The CPU path is the C row engine, the CUDA path is the shared-memory tile kernel. At 2^27
elements the tile reads a little below its 2^28 rate because launch and clock ramp are a
larger share of the time.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
