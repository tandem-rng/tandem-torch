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
| `Tandem.randn` float32 / float64, 2^22, minimum of 5 | 5.38 / 4.84 |
| `torch.randn` float32 / float64, 2^22, minimum of 5 | 0.55 / 1.02 |
| `Tandem.exponential` float32 / float64, 2^22, minimum of 5 | 6.51 / 5.95 |
| `Tensor.exponential_` float32 / float64, 2^22, minimum of 5 | 0.52 / 1.03 |

## GPU

NVIDIA A100 40 GB PCIe, GPU idle, cudaEvent timings, 0.5 s warm-up, minimum of 21:

| | GiB/s |
|---|---|
| `Tandem.rand` float32 / float64 | 1360 / 1381 |
| `torch.rand` float32 / float64 | 1242 / 1313 |
| `Tandem.bits` uint32 | 1364 |
| `Tandem` fill uint8 / bool / float16 | 1272 / 1090 / 1306 |
| `Tandem.randint` int32 `[0, 1000)`, `out=` / allocating | 1316 / 1316 |
| `torch.randint` int32 `[0, 1000)` | 909 |
| `Tandem.randint` int32 full range | 1309 |
| `torch.randint` int32 full range | 324 |
| `Tandem.randint` int64 `[0, 1000)` | 1351 |
| `torch.randint` int64 `[0, 1000)` | 1306 |
| `Tandem.randint` int64 `[-2^62, 2^62)` | 1338 |
| `torch.randint` int64 `[-2^62, 2^62)` | 632 |
| `Tandem.randn` float32 `out=` / allocating | 1255 / 1265 |
| `torch.randn` float32 | 913 |
| `Tandem.randn` float64 `out=` / allocating | 908 / 983 |
| `torch.randn` float64 | 587 |
| `Tandem.exponential` float32 / float64 | 986 / 913 |
| `Tensor.exponential_` float32 / float64 | 1003 / 562 |

`randint` and `randn` take `out=` and then write into it, otherwise they allocate. An empty
`out` takes the requested shape, as in torch. A nonempty `out` of another shape is an error,
since torch deprecates resizing it. On CUDA,
`randint` into int32, uint32, int64 or uint64 adds the low bound and widens inside the bounded
kernel (`tandem::fill_u32_below` and `fill_u64_below` with a low bound), so there is no second
pass over the output. Other dtypes, and a range of exactly 2^32 or 2^64, take the unfused path.
