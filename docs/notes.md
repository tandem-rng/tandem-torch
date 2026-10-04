# tandem-torch notes

Material moved out of the README.

## Overview

This is not a `torch.Generator`. That class is final and its RNG hooks are internal to
PyTorch, so no third-party generator can drive `torch.rand`. `tandem_torch` fills tensors
through its own functions instead, which is also what makes the stream reproducible across
languages and devices.

## Install

The reference sources sit in the `external/tandem-c` and `external/tandem-cuda` git
submodules. Clone with `git clone --recurse-submodules`, or run `git submodule update --init`
in an existing clone. GitHub's ZIP download omits submodules and does not build.

The build needs a C and C++ compiler. With `nvcc` on the path (or `CUDA_HOME` set) the CUDA
fills are built too, otherwise CUDA tensors raise. Set `TANDEM_TORCH_CUDA=0` or `1` to force
the choice. For development, `pixi install` gives a CPU environment, `pixi install -e cuda`
a Linux GPU environment with nvcc 12.8 from conda-forge and PyTorch's cu128 wheel, and
`pixi run test` runs the tests.

## What it provides

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
on CPU, `tandem::fill_normal_f64` and `_f32` on CUDA). The CPU fills and the CUDA float64
fill share one polynomial log, sin and cos with explicit fused multiply-adds, so they equal
tandem-c bit for bit. The CUDA float32 fill uses `logf` and the fast `__sincosf` and agrees to
16 ulps. `float16` and `bfloat16`
round the float32 normal. Empty bounded and normal fills leave the position alone.

`exponential` is `-log(1 - u)` of uniform draw i for element i, with the polynomial log of the
normals (`tandem_fill_exponential_f64` and `_f32` on CPU, `tandem::fill_exponential_f64` and
`_f32` on CUDA). Both devices equal tandem-c bit for bit, float32 included. `float16` and
`bfloat16` round the float32 value, and an empty fill leaves the position alone. It is not in
the specification.

`randint(low, high, size, dtype=torch.int64, device="cpu")` draws on `[low, high)`. A range of
at most 2^32 takes one 32-bit draw per element (`tandem_fill_u32_below` on CPU,
`tandem::fill_u32_below` on CUDA), a larger range one 64-bit draw. Both use Lemire's method and
a rejected draw retries on a fallback stream keyed by its global draw index, so a fill cut at
any element equals the whole fill and CPU and CUDA give the same values and every
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

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills from several offsets with the reference
stream dumps in `tests/data`, complex types included. It checks `randint`, `randn` and
`exponential` against
`tests/cross.json`, which `tools/cross_json.py` makes from the cross-check headers of the
submodules and which holds the values of tandem-c and tandem-cuda, `randperm` against a
Python Fisher-Yates over the stream words, `bfloat16` against the 16-bit word fill and `at`
against fills. With a CUDA device the suite also runs the cross-checks there and compares CUDA
fills with CPU fills for every dtype, four chunk lengths, fourteen positions and nine lengths,
and on storage that is not 16-byte aligned. CI runs the CPU tests on Linux and macOS and fails
when the vectors drift from upstream or a submodule pin is not on its upstream main.
`tools/bump.sh` moves the pins to the latest main. The CUDA tests run by hand on a GPU host.

## Speed

`randint` and `randn` take `out=` and then write into it, otherwise they allocate. An empty
`out` takes the requested shape, as in torch. A nonempty `out` of another shape is an error,
since torch deprecates resizing it. On CUDA,
`randint` into int32, uint32, int64 or uint64 adds the low bound and widens inside the bounded
kernel (`tandem::fill_u32_below` and `fill_u64_below` with a low bound), so there is no second
pass over the output. Other dtypes, and a range of exactly 2^32 or 2^64, take the unfused path.
