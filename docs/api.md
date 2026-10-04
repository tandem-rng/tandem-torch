# API

```python
import torch
from tandem_torch import Tandem

t = Tandem(42)                                   # the spec's stream for seed 42
u = t.rand(1_000_000)                            # float64 in [0, 1), 53 random bits
f = t.rand(1 << 20, dtype=torch.float32, device="cuda")
w = t.bits(1 << 20, dtype=torch.uint32)          # stream words
z = t.randn(1000)                                # Box-Muller, two float64 uniforms each
e = t.exponential(1000)                          # -log(1 - u), one float64 uniform each
b = t.randbool(100)
i = t.randint(-5, 5, (4, 4))                     # int64 on [-5, 5), Lemire, same values on CUDA
p = t.randperm(10)                               # Fisher-Yates, defined on the CPU
x = t.shuffle(torch.arange(12).view(3, 4), dim=1)
t.fill_(torch.empty(4096, dtype=torch.float16))  # fill in place, any supported dtype
x = t.at(torch.float32, 1000)                    # element 1000 of the next float32 fill, no advance
worker = t.split(7)                              # by index, from the key alone
kids = t.fork(4)                                 # from the current block, parent moves on
t.key, t.position, t.chunk_length                # transport form
s = t.get_state(); t.set_state(s)                # as on torch.Generator, also manual_seed
```

- `Tandem(seed)`, `Tandem.from_key(key, position, K)`: the generator, with `key`, `position`
  and `chunk_length`. `get_state`, `set_state` and `manual_seed` as on `torch.Generator`.
- `rand`, `bits`, `randbool`, `fill_`: `bool`, `uint8` to `uint64`, signed integers, `float16`,
  `float32`, `float64`, `complex64` and `complex128`.
- `rand` with `bfloat16`: an extension, `(raw16 >> 8) * 2^-8`, not in the specification.
- `randint(low, high, size, dtype, device, out=)`: Lemire's method, same values on CUDA.
- `randn(*shape, out=)`: Box-Muller from tandem-cuda. Bit exact except CUDA float32, within 16 ulps.
- `exponential(*shape, out=)`: `-log(1 - u)` as tandem-c, bit exact on CPU and CUDA.
- `randperm(n)`, `shuffle(x, dim)`: Fisher-Yates, defined on the CPU, `n < 2^32`.
- `at(dtype, i)`: element `i` of the next fill without advancing, for 32 and 64-bit types.
- `split(index)`, `fork(n)`, `sub(purpose)`: child generators.
- Parallel use: ranks that start at the position of their first element reproduce a serial run.

Bounded draws, normals, exponentials, `randperm` and `bfloat16` are not in the specification.

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
