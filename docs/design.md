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

`randn` follows Appendix A of the specification (`tandem_fill_normal_f64` and `_f32` on CPU,
`tandem::fill_normal_f64` and `_f32` on CUDA).

float64 is the 1024-layer ziggurat. Element i comes from 64-bit draw i, so a fill cut at any
element equals the whole fill. A draw outside the inner rectangles, about 0.4 %, continues on a
fallback stream keyed by its global draw index. Both devices equal tandem-c bit for bit. An empty
float64 fill aligns the position to 64, as section 5 of the specification says. On CUDA the fill
takes its miss list from `cudaMallocAsync`, so the extension raises the release threshold of the
device's default memory pool. Without that, every fill maps fresh memory and runs at half speed.

float32 is Box-Muller. Elements 2j and 2j + 1 are the cos and sin halves from the float32
uniforms 2j and 2j + 1, so an odd count still consumes both uniforms of its last pair. The CPU
equals tandem-c bit for bit. The CUDA fill uses `logf` and the fast `__sincosf` and agrees to 16
ulps. `float16` and `bfloat16` round the float32 normal. Empty bounded and float32 normal fills
leave the position alone.

## Exponentials

`exponential` is `-log(1 - u)` of uniform draw i for element i, with the polynomial log of the
normals (`tandem_fill_exponential_f64` and `_f32` on CPU, `tandem::fill_exponential_f64` and
`_f32` on CUDA). Both devices equal tandem-c bit for bit, float32 included. `float16` and
`bfloat16` round the float32 value, and an empty fill leaves the position alone. It is not in
the specification.
