# API

## Use

```python
import torch
from tandem_torch import Tandem

t = Tandem(42)                                   # the spec's stream for seed 42
u = t.rand(1_000_000)                            # float64 in [0, 1), 53 random bits
f = t.rand(1 << 20, dtype=torch.float32, device="cuda")
w = t.bits(1 << 20, dtype=torch.uint32)          # stream words
z = t.randn(1000)                                # ziggurat, one 64-bit draw each
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

## Reference

- `Tandem(seed)`, `Tandem.from_key(key, position, K)`: the generator, with `key`, `position`
  and `chunk_length`. `get_state`, `set_state` and `manual_seed` as on `torch.Generator`.
- `rand`, `bits`, `randbool`, `fill_`: `bool`, `uint8` to `uint64`, signed integers, `float16`,
  `float32`, `float64`, `complex64` and `complex128`.
- `rand` with `bfloat16`: an extension, `(raw16 >> 8) * 2^-8`, not in the specification.
- `randint(low, high, size, dtype, device, out=)`: Lemire's method, same values on CUDA.
- `randn(*shape, out=)`: float64 ziggurat, float32 Box-Muller. Bit exact except CUDA float32, within 16 ulps.
- `exponential(*shape, out=)`: `-log(1 - u)` as tandem-c, bit exact on CPU and CUDA.
- `randperm(n)`, `shuffle(x, dim)`: Fisher-Yates, defined on the CPU, `n < 2^32`.
- `at(dtype, i)`: element `i` of the next fill without advancing, for 32 and 64-bit types.
- `split(index)`, `fork(n)`, `sub(purpose)`: child generators.

Bounded draws, normals, exponentials, `randperm` and `bfloat16` are not in the specification.
[Design](design.md) gives their contracts.

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

`at(dtype, i)` is random access: element `i` of the fill of `dtype` that would start at the
current position, as a Python number, computed on the host without advancing. It supports
`int32`, `uint32`, `int64`, `uint64`, `float32` and `float64`, the types random access has in
tandem-c and tandem-cuda.

## Parallel use

Element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.
