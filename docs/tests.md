# Tests

```sh
pixi run test     # tests/test_tandem.py
```

## Suite

- Every vector of the specification, from `tests/vectors.json`.
- Stream dumps in `tests/data` from several offsets.
- `randint`, `randn` and `exponential` against `tests/cross.json`, made by `tools/cross_json.py` from the
  submodule fixtures. `tools/bump.sh` moves the pins to the latest main.
- With a CUDA device, CUDA fills against CPU fills for every dtype.

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills from several offsets with the reference
stream dumps in `tests/data`, complex types included. It checks `randint`, `randn` and
`exponential` against
`tests/cross.json`, which `tools/cross_json.py` makes from the cross-check headers of the
submodules and which holds the values of tandem-c and tandem-cuda, `randperm` against a
Python Fisher-Yates over the stream words, `bfloat16` against the 16-bit word fill and `at`
against fills. With a CUDA device the suite also runs the cross-checks there and compares CUDA
fills with CPU fills for every dtype, four chunk lengths, fourteen positions and nine lengths,
and on storage that is not 16-byte aligned.

## Fixtures

`tools/bump.sh` moves the pins to the latest main.

## CI

CI runs the CPU tests on Linux and macOS and fails when the vectors drift from upstream or a
submodule pin is not on its upstream main. The CUDA tests run by hand on a GPU host.
