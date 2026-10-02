"""Agreement with the specification vectors and the Julia dumps, on CPU and, if present, CUDA."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import tandem_torch as tt

HERE = Path(__file__).parent
VEC = json.loads((HERE / "vectors.json").read_text())
KEY = tuple(int(w, 16) for w in VEC["key"])
K = VEC["K"]
CUDA = pytest.mark.skipif(not (torch.cuda.is_available() and tt.has_cuda), reason="no CUDA")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() and tt.has_cuda else [])


def words(ws):
    return tuple(int(w, 16) for w in ws)


def dump(name, dtype):
    return torch.from_numpy(np.fromfile(HERE / "data" / name, dtype=dtype).copy())


def make(name):
    if name.startswith("k1234_K8_"):
        return tt.Tandem.from_key((1, 2, 3, 4), 0, 8)
    if name.startswith("k1234_"):
        return tt.Tandem.from_key((1, 2, 3, 4), 0, 32)
    return tt.Tandem(42)


# ---- Specification vectors ---------------------------------------------------------------


def test_stream_words():
    t = tt.Tandem.from_key(KEY, 0, K)
    got = t.bits(64, dtype=torch.uint32)
    for s in VEC["stream_words"]:
        i = s["first_word"]
        assert tuple(got[i:i + 4].tolist()) == words(s["words"])
    assert t.position == 64 * 32


def test_draws_from_position_0():
    d = VEC["draws_from_position_0"]
    f64 = tt.Tandem.from_key(KEY, 0, K).rand(32)
    for i, x in d["Float64"].items():
        assert f64[int(i)].item() == x
    f32 = tt.Tandem.from_key(KEY, 0, K).rand(8, dtype=torch.float32)
    for i, x in d["Float32"].items():
        assert f32[int(i)].item() == pytest.approx(x, abs=0) or np.float32(f32[int(i)].item()) == np.float32(x)
    b = tt.Tandem.from_key(KEY, 0, K).randbool(129)
    for i, x in d["Bool"].items():
        assert b[int(i)].item() == bool(x)


def test_derived_keys():
    k = VEC["derived_keys"]
    t = tt.Tandem.from_key(KEY, 0, K)
    assert t.split(0).key == words(k["split_child_0"])
    assert t.split(1).key == words(k["split_child_1"])
    assert t.sub(7).key == words(k["purpose_7"])
    kids = t.fork(2)
    assert kids[0].key == words(k["fork_child_0_at_block_0"])
    assert kids[0].position == 0 and kids[0].chunk_length == K
    assert t.position == 128


def test_seed_whitening():
    s = VEC["seed_whitening"]
    t = tt.Tandem(s["seed"])
    assert t.key == words(s["key"])
    f64 = tt.Tandem(s["seed"]).rand(32)
    for i, x in s["Float64"].items():
        assert f64[int(i)].item() == x
    u32 = tt.Tandem(s["seed"]).bits(4, dtype=torch.uint32)
    for i, x in s["UInt32"].items():
        assert u32[int(i)].item() == int(x, 16)


# ---- Julia dumps -------------------------------------------------------------------------

DUMPS = [
    ("k1234_K32_u32.bin", np.uint32, torch.uint32),
    ("k1234_K8_u32.bin", np.uint32, torch.uint32),
    ("k1234_K32_u64.bin", np.uint64, torch.uint64),
    ("seed42_K32_f64.bin", np.float64, torch.float64),
    ("seed42_K32_f32.bin", np.float32, torch.float32),
    ("seed42_K32_u8.bin", np.uint8, torch.uint8),
    ("seed42_K32_f16bits.bin", np.uint16, torch.float16),
    ("seed42_K32_bool.bin", np.uint8, torch.bool),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,npdtype,dtype", DUMPS)
def test_dumps(name, npdtype, dtype, device):
    want = dump(name, npdtype)
    if dtype == torch.float16:
        want = want.view(torch.float16)
    if dtype == torch.bool:
        want = want.bool()
    n = want.numel()
    t = make(name)
    got = t.fill_(torch.empty(n, dtype=dtype, device=device)).cpu()
    assert torch.equal(got, want), name
    w = 1 if dtype == torch.bool else got.element_size() * 8
    assert t.position == n * w
    # Resuming from any offset gives the rest of the dump.
    for start in (1, 7, 33, 100, 257):
        t = make(name)
        t.position = start * w
        part = t.fill_(torch.empty(n - start, dtype=dtype, device=device)).cpu()
        assert torch.equal(part, want[start:]), (name, start)


def test_signed_types_reinterpret():
    u = tt.Tandem(3).bits(1000, dtype=torch.uint32)
    s = tt.Tandem(3).bits(1000, dtype=torch.int32)
    assert torch.equal(u.view(torch.int32), s)


def test_mixed_widths_align():
    t = tt.Tandem(42)
    t.bits(1, dtype=torch.uint8)
    assert t.position == 8
    x = t.rand(1)
    assert t.position == 128
    assert x.item() == tt.Tandem(42).rand(2)[1].item()


def test_shapes_and_out():
    t = tt.Tandem(5)
    a = t.rand(3, 4)
    assert a.shape == (3, 4)
    buf = torch.empty(12)
    assert tt.Tandem(5).rand(out=buf) is buf
    assert torch.equal(buf, tt.Tandem(5).rand(12, dtype=torch.float32))
    with pytest.raises(TypeError):
        tt.Tandem(5).rand(out=buf, dtype=torch.float64)
    nc = torch.empty(4, 6)[:, ::2]
    assert not nc.is_contiguous()
    tt.Tandem(5).fill_(nc)
    assert torch.equal(nc.flatten(), tt.Tandem(5).rand(12, dtype=torch.float32))


def test_randn():
    z = tt.Tandem(9).randn(200_000)
    assert torch.isfinite(z).all()
    assert abs(z.mean().item()) < 0.01
    assert abs(z.std().item() - 1) < 0.01
    assert tt.Tandem(9).randn(5, dtype=torch.float32).dtype == torch.float32


def test_functional_forms():
    x, nxt = tt.rand(KEY, 0, 16)
    assert nxt == 1024
    y, nxt2 = tt.rand(KEY, nxt, 16)
    whole = tt.Tandem.from_key(KEY).rand(32)
    assert torch.equal(torch.cat([x, y]), whole) and nxt2 == 2048
    b, _ = tt.bits(KEY, 0, 4, dtype=torch.uint32)
    assert b.dtype == torch.uint32


def test_pickle_and_eq():
    t = tt.Tandem(42)
    t.rand(3)
    u = pickle.loads(pickle.dumps(t))
    assert u == t and hash(u) == hash(t)
    assert repr(t).startswith("Tandem(key=")


def test_argument_errors():
    with pytest.raises(ValueError):
        tt.Tandem(-1)
    with pytest.raises(ValueError):
        tt.Tandem(0, K=3)
    with pytest.raises(TypeError):
        tt.Tandem(0).rand(2, dtype=torch.int32)
    with pytest.raises(TypeError):
        tt.Tandem(0).bits(2, dtype=torch.float32)
    with pytest.raises(TypeError):
        tt.Tandem(0).fill_(torch.empty(2, dtype=torch.bfloat16))


def test_entropy_seeds_differ():
    assert tt.Tandem().key != tt.Tandem().key


# ---- CUDA agrees with the CPU ------------------------------------------------------------

ALL_DTYPES = [torch.bool, torch.uint8, torch.int8, torch.uint16, torch.int16, torch.float16,
              torch.uint32, torch.int32, torch.float32, torch.uint64, torch.int64, torch.float64]


@CUDA
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("K", [1, 8, 32, 128])
def test_cuda_equals_cpu(dtype, K):
    for pos in (0, 1, 8, 16, 31, 33, 64, 127, 128, 1000, 1024, 1025, 4096 * 7 + 3, 2**20 + 5):
        for n in (0, 1, 3, 16, 31, 32, 33, 1000, 2**16 + 7):
            key = (0xdeadbeef, pos & 0xffffffff, K, n)
            a, na = tt.bits(key, pos, n, dtype=dtype, device="cpu", K=K) if dtype not in (
                torch.float16, torch.float32, torch.float64) else tt.rand(key, pos, n, dtype=dtype, K=K)
            b, nb = tt.bits(key, pos, n, dtype=dtype, device="cuda", K=K) if dtype not in (
                torch.float16, torch.float32, torch.float64) else tt.rand(key, pos, n, dtype=dtype, device="cuda", K=K)
            assert na == nb
            assert torch.equal(a, b.cpu()), (dtype, K, pos, n)


@CUDA
def test_cuda_unaligned_storage():
    big = torch.empty(2**16 + 9, dtype=torch.uint32, device="cuda")
    view = big[1:]  # 4 bytes off a 16-byte boundary
    tt.Tandem(7).fill_(view)
    assert torch.equal(view.cpu(), tt.Tandem(7).bits(2**16 + 8, dtype=torch.uint32))
