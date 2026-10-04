<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .pt"></p>

# tandem-torch

PyTorch tensors from [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
random number generator. CPU and CUDA fills write the specification's stream bit for bit, fast
on both. It is not a `torch.Generator`, which third-party code cannot implement.

## Install

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-torch
pip install torch                      # a CPU or CUDA build, see pytorch.org
pip install --no-build-isolation .
```

The build needs a C and C++ compiler, and builds the CUDA fills when `nvcc` is on the path or
`CUDA_HOME` is set. `TANDEM_TORCH_CUDA=0` or `1` forces the choice. The submodules
`external/tandem-c` (2b6e075) and `external/tandem-cuda` (b65745a) hold the reference sources.
`pixi install` gives a CPU environment, `pixi install -e cuda` a Linux GPU environment.

Full notes on dtypes, normals, bounded draws, tests and speed: [docs/notes.md](docs/notes.md).

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

Every call aligns the position to the element width, reads, and advances, as the specification
requires. The functional forms `tandem_torch.rand(key, position, *shape, ...)` and
`tandem_torch.bits(...)` return `(tensor, next_position)`.

## What it provides

- `Tandem(seed)`, `Tandem.from_key(key, position, K)`: the generator, with `key`, `position`
  and `chunk_length`.
- `rand`, `bits`, `randbool`, `fill_`: `bool`, `uint8` to `uint64`, signed integers, `float16`,
  `float32`, `float64`, `complex64` and `complex128`.
- `rand` with `bfloat16`: an extension, `(raw16 >> 8) * 2^-8`, not in the specification.
- `randint(low, high, size, dtype, device, out=)`: Lemire's method, same values on CUDA.
- `randn(*shape, out=)`: Box-Muller from tandem-cuda. Devices agree to a few ulps.
- `randperm(n)`, `shuffle(x, dim)`: Fisher-Yates, defined on the CPU, `n < 2^32`.
- `at(dtype, i)`: element `i` of the next fill without advancing, for 32 and 64-bit types.
- `split(index)`, `fork(n)`, `sub(purpose)`: child generators.
- Parallel use: ranks that start at the position of their first element reproduce a serial run.

Bounded draws, normals, `randperm` and `bfloat16` are not in the specification.

## Tests

`pixi run test` runs `tests/test_tandem.py`.

- Every vector of the specification, from `tests/vectors.json`.
- Stream dumps in `tests/data` from several offsets.
- `randint` and `randn` against `tests/cross.json`, made by `tools/cross_json.py` from the
  submodule fixtures. `tools/bump.sh` moves the pins to the latest main.
- With a CUDA device, CUDA fills against CPU fills for every dtype.

## Speed

`python tools/bench.py [cpu|cuda]`, preallocated outputs, 2^27 elements.

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
| `Tandem.randint` int32 `[0, 1000)`, `out=` / allocating | 1285 / 1268 |
| `torch.randint` int32 `[0, 1000)` | 839 |
| `Tandem.randint` int32 full range | 1252 |
| `torch.randint` int32 full range | 332 |
| `Tandem.randint` int64 `[0, 1000)` | 1331 |
| `torch.randint` int64 `[0, 1000)` | 1282 |
| `Tandem.randint` int64 `[-2^62, 2^62)` | 1255 |
| `torch.randint` int64 `[-2^62, 2^62)` | 621 |
| `Tandem.randn` float32 `out=` / allocating | 1138 / 1125 |
| `torch.randn` float32 | 774 |
| `Tandem.randn` float64 `out=` / allocating | 705 / 693 |
| `torch.randn` float64 | 583 |

`randint` and `randn` write into `out=` when given, otherwise they allocate.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
