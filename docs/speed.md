# Speed

`python tools/bench.py [cpu|cuda]` produces the figures, with preallocated outputs and 2^27 elements.

## CPU

Apple M4, one thread, minimum of 7:

| | GiB/s |
|---|---|
| `Tandem.rand` float32 / float64 | 17.4 / 17.6 |
| `torch.rand` float32 / float64 | 3.3 / 4.8 |
| `Tandem.bits` uint32 | 20.2 |
| `torch.randint` int32 | 2.3 |
| `Tandem.randn` float32 / float64, 2^22, minimum of 5 | 5.54 / 7.71 |
| `torch.randn` float32 / float64, 2^22, minimum of 5 | 0.55 / 1.02 |
| `Tandem.exponential` float32 / float64, 2^22, minimum of 5 | 6.51 / 5.95 |
| `Tensor.exponential_` float32 / float64, 2^22, minimum of 5 | 0.52 / 1.03 |

## GPU

NVIDIA A100 40 GB PCIe, cudaEvent timings, 0.5 s warm-up, minimum of 21, `python tools/bench.py
cuda`. The GPU had no other process, and a second run agreed within 5 %. The third-party columns
come from the same run, by the same method: cuRAND Philox4x32-10 of cuRAND 10.3.9 on torch's
stream through the libcurand of PyTorch's CUDA wheel, then torch's own generator into a
preallocated tensor. cuRAND has no 8-, 16- or 64-bit integer output for Philox, no bounded
integers and no exponentials, so the cuRAND column gives the nearest call for those rows, marked
"nearest": `curandGenerate` into the same bytes, or the uniform the exponential reads.

| | Tandem | cuRAND Philox4x32-10 | cuRAND call | torch |
|---|---|---|---|---|
| `rand` float32 / float64 | 1320 / 1355 | 1262 / 789 | `curandGenerateUniform`, `curandGenerateUniformDouble` | 1105 / 1179 |
| `bits` uint32 | 1323 | 1285 | `curandGenerate` | |
| fill uint8 / bool / float16 | 1152 / 984 / 1203 | 1197 | `curandGenerate`, nearest | |
| `randint` int32 `[0, 1000)`, `out=` / allocating | 1278 / 1242 | 1285 | `curandGenerate`, nearest | 836 |
| `randint` int32 full range, `out=` / allocating | 1272 / 1239 | 1285 | `curandGenerate`, nearest | 328 |
| `randint` int64 `[0, 1000)`, `out=` / allocating | 1336 / 1311 | 1294 | `curandGenerate`, nearest | 1288 |
| `randint` int64 `[-2^62, 2^62)`, `out=` / allocating | 1282 / 1275 | 1294 | `curandGenerate`, nearest | 615 |
| `randn` float32, `out=` / allocating | 1165 / 1120 | 852 | `curandGenerateNormal` | 773 |
| `randn` float64, `out=` / allocating | 1026 / 1013 | 560 | `curandGenerateNormalDouble` | 568 |
| `exponential` float32 / float64 (torch: `Tensor.exponential_`) | 899 / 903 | 1262 / 789 | `curandGenerateUniform`, `curandGenerateUniformDouble`, nearest | 1011 / 562 |

cuRAND leads where its nearest call does less work: 32-bit words where the fill stores bytes,
bools or halves, no bounding, and uniforms without the logarithm. The table was measured with
tandem-cuda e98daee. Its float32 exponential is within 0.571 ulp for every draw, so that
`1 - exp(-x)` maps back to the draw's own 2^-24 grid point. Its two-float logarithm takes 10 more
f32 operations per draw than the single-float one before it, and torch's `Tensor.exponential_`
leads that cell.

`randint` and `randn` take `out=` and then write into it, otherwise they allocate. An empty
`out` takes the requested shape, as in torch. A nonempty `out` of another shape is an error,
since torch deprecates resizing it. On CUDA,
`randint` into int32, uint32, int64 or uint64 adds the low bound and widens inside the bounded
kernel (`tandem::fill_u32_below` and `fill_u64_below` with a low bound), so there is no second
pass over the output. Other dtypes, and a range of exactly 2^32 or 2^64, take the unfused path.
