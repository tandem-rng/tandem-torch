# Design

## Fills

On CUDA every dtype has its own fill kernel in `tandem.cuh`, including `bool`, the 8-bit and
16-bit types and `float16`. `bfloat16` is the 16-bit word fill followed by the scaling.

## Bounded integers

`randint(low, high, size, dtype=torch.int64, device="cpu")` draws on `[low, high)`. A range of
at most 2^32 takes one 32-bit draw per element (`tandem_fill_u32_below` on CPU,
`tandem::fill_u32_below` on CUDA), a larger range one 64-bit draw. Both use Lemire's method and
a rejected draw retries on a fallback stream keyed by its global draw index, so a fill cut at
any element equals the whole fill and CPU and CUDA give the same values and every
element consumes exactly one draw. `randperm(n)` is Fisher-Yates from the end with one
sequential scalar bounded draw per swap, `tandem_u32_below` over the stream, so the CPU
defines it and a CUDA result is copied from there. `shuffle(x, dim=0)` indexes `x` with a
`randperm`. None of these is in the specification, and `randperm` needs `n < 2^32`.

## Normals

`randn` is Box-Muller as in tandem-cuda, not part of the specification. Elements 2j and 2j + 1
are the cos and sin halves of one step from the uniforms 2j and 2j + 1 of the plain float fill,
so an odd count still consumes both uniforms of its last pair. A float64 pair takes 128 stream
bits and a float32 pair 64, with float32 computed in float (`tandem_fill_normal_f64` and `_f32`
on CPU, `tandem::fill_normal_f64` and `_f32` on CUDA). The CPU fills and the CUDA float64
fill share one polynomial log, sin and cos with explicit fused multiply-adds, so they equal
tandem-c bit for bit. The CUDA float32 fill uses `logf` and the fast `__sincosf` and agrees to
16 ulps. `float16` and `bfloat16`
round the float32 normal. Empty bounded and normal fills leave the position alone.

## Exponentials

`exponential` is `-log(1 - u)` of uniform draw i for element i, with the polynomial log of the
normals (`tandem_fill_exponential_f64` and `_f32` on CPU, `tandem::fill_exponential_f64` and
`_f32` on CUDA). Both devices equal tandem-c bit for bit, float32 included. `float16` and
`bfloat16` round the float32 value, and an empty fill leaves the position alone. It is not in
the specification.
