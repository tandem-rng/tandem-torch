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
cuda`. The GPU had no other process, and a second run agreed within 3 %. The cuRAND rows are
Philox4x32-10 of cuRAND 10.3.9 in the same run, by the same method, on torch's stream through the
libcurand of PyTorch's CUDA wheel. cuRAND has no 8-, 16- or 64-bit integer output for Philox, no
bounded integers and no exponentials, so the cuRAND column gives the nearest call for those rows,
marked "nearest": `curandGenerate` into the same bytes, or the uniform the exponential reads.

| | GiB/s | cuRAND Philox4x32-10 |
|---|---|---|
| `Tandem.rand` float32 / float64 | 1327 / 1364 | 1259 / 793 (`curandGenerateUniform`, `curandGenerateUniformDouble`) |
| `torch.rand` float32 / float64 | 1110 / 1195 | |
| `Tandem.bits` uint32 | 1327 | 1278 (`curandGenerate`) |
| `Tandem` fill uint8 / bool / float16 | 1174 / 992 / 1221 | 1209, nearest (`curandGenerate`) |
| `Tandem.randint` int32 `[0, 1000)`, `out=` / allocating | 1282 / 1255 | 1278, nearest (`curandGenerate`) |
| `torch.randint` int32 `[0, 1000)` | 839 | |
| `Tandem.randint` int32 full range, `out=` / allocating | 1275 / 1249 | 1278, nearest (`curandGenerate`) |
| `torch.randint` int32 full range | 336 | |
| `Tandem.randint` int64 `[0, 1000)`, `out=` / allocating | 1341 / 1325 | 1302, nearest (`curandGenerate`) |
| `torch.randint` int64 `[0, 1000)` | 1292 | |
| `Tandem.randint` int64 `[-2^62, 2^62)`, `out=` / allocating | 1280 / 1278 | 1302, nearest (`curandGenerate`) |
| `torch.randint` int64 `[-2^62, 2^62)` | 619 | |
| `Tandem.randn` float32 `out=` / allocating | 1123 / 1112 | 864 (`curandGenerateNormal`) |
| `torch.randn` float32 | 774 | |
| `Tandem.randn` float64 `out=` / allocating | 1028 / 1027 | 572 (`curandGenerateNormalDouble`) |
| `torch.randn` float64 | 569 | |
| `Tandem.exponential` float32 / float64 | 984 / 900 | 1259 / 793, nearest (`curandGenerateUniform`, `curandGenerateUniformDouble`) |
| `Tensor.exponential_` float32 / float64 | 999 / 562 | |

cuRAND leads where its nearest call does less work: 32-bit words where the fill stores bytes,
bools or halves, no bounding, and uniforms without the logarithm. The extension pins tandem-cuda
6ad0817, before its folded exponential, which tandem-cuda runs at 1149 GiB/s in float32.

`randint` and `randn` take `out=` and then write into it, otherwise they allocate. An empty
`out` takes the requested shape, as in torch. A nonempty `out` of another shape is an error,
since torch deprecates resizing it. On CUDA,
`randint` into int32, uint32, int64 or uint64 adds the low bound and widens inside the bounded
kernel (`tandem::fill_u32_below` and `fill_u64_below` with a low bound), so there is no second
pass over the output. Other dtypes, and a range of exactly 2^32 or 2^64, take the unfused path.
