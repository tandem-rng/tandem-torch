"""Agreement with the specification vectors and the Julia dumps, on CPU and, if present, CUDA."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import tandem_torch as tt

HERE = Path(__file__).parent
VEC = json.loads((HERE / "vectors.json").read_text())
CROSS = json.loads((HERE / "cross.json").read_text())
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
    ("seed42_K32_c32.bin", np.float32, torch.complex64),
    ("seed42_K32_c64.bin", np.float64, torch.complex128),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,npdtype,dtype", DUMPS)
def test_dumps(name, npdtype, dtype, device):
    want = dump(name, npdtype)
    if dtype == torch.float16:
        want = want.view(torch.float16)
    if dtype == torch.bool:
        want = want.bool()
    if dtype.is_complex:
        want = want.view(dtype)
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


@pytest.mark.parametrize("dtype", [torch.uint32, torch.int32, torch.uint64, torch.int64,
                                   torch.float32, torch.float64])
def test_at_is_element_i_of_the_fill_without_advancing(dtype):
    t = tt.Tandem(6, K=8)
    t.position = 37  # not aligned to any width, so the alignment is part of the contract
    idx = (0, 1, 31, 32, 1000, 4097)
    want = tt.Tandem.from_key(t.key, 37, 8).fill_(torch.empty(4098, dtype=dtype))
    assert [t.at(dtype, i) for i in idx] == [want[i].item() for i in idx]
    assert t.position == 37


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


def test_complex_aligns_by_component():
    t = tt.Tandem(42)
    t.bits(1, dtype=torch.uint8)
    z = t.rand(1, dtype=torch.complex64)
    assert t.position == 32 + 64
    assert torch.view_as_real(z).tolist() == [tt.Tandem(42).rand(3, dtype=torch.float32)[1:].tolist()]


@pytest.mark.parametrize("device", DEVICES)
def test_bfloat16_is_the_scaled_top_byte_of_the_u16_word(device):
    t = tt.Tandem(3)
    got = t.rand(1000, dtype=torch.bfloat16, device=device).cpu()
    words = tt.Tandem(3).bits(1000, dtype=torch.uint16)
    assert torch.equal(got.float(), (words.to(torch.int32) >> 8).float() * 2.0**-8)
    assert t.position == 16000


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_randn_moments(device, dtype):
    z = tt.Tandem(9).randn(200_000, dtype=dtype, device=device)
    assert z.dtype == dtype and torch.isfinite(z).all()
    assert abs(z.mean().item()) < 0.01
    assert abs(z.std().item() - 1) < 0.01


def same_normals(got, want):
    """Bit for bit, except float32 on CUDA, whose fast sincos is within 16 ulps and 1e-6."""
    want = torch.tensor(want, dtype=got.dtype)
    if got.dtype == torch.float32 and got.is_cuda:
        got, want = got.cpu().double(), want.double()
        return bool(((got - want).abs() <= 16 * 2.0**-23 * want.abs() + 1e-6).all())
    return torch.equal(got.cpu(), want)


@pytest.mark.parametrize("device", DEVICES)
def test_randn_matches_the_other_ports(device):
    """The pairs of tandem-c's Box-Muller after one bool, and tandem-cuda's fills from the key
    of seed 42 at several positions, odd counts included."""
    for dtype, want, end in ((torch.float64, CROSS["normal_f64"], CROSS["normal_f64_end_pos"]),
                             (torch.float32, CROSS["normal_f32"], CROSS["normal_f32_end_pos"])):
        t = tt.Tandem(42)
        t.randbool(1)
        assert same_normals(t.randn(len(want), dtype=dtype, device=device), want), dtype
        assert t.position == end
    key = tuple(CROSS["cuda_key"])
    for dtype, rows, w in ((torch.float64, CROSS["cuda_normal64"], 64), (torch.float32, CROSS["cuda_normal32"], 32)):
        for row in rows:
            z, nxt = tt.randn(key, row["pos"], row["n"], dtype=dtype, device=device)
            assert same_normals(z, row["out"]), (dtype, row["pos"])
            assert nxt == -(-row["pos"] // w) * w + (row["n"] + 1) // 2 * 2 * w


def test_randn_bits_match_tandem_c():
    """The bytes of tandem-c's tools/dump_normals.c: 2e6 - 1 float64 then float32 normals from
    five positions. tandem-c records their FNV-1a hash 0x9414e1315e2653be, this test SHA-256."""
    h = hashlib.sha256()
    key = tt.Tandem(2026 + (7 << 64)).key
    for start in (0, 1, 77, 12345, 1 << 30):
        z, nxt = tt.randn(key, start, 2_000_000 - 1)
        h.update(z.numpy().tobytes())
        h.update(tt.randn(key, nxt, 2_000_000 - 1, dtype=torch.float32)[0].numpy().tobytes())
    assert h.hexdigest() == "cfae418807a7d5f91ecd3e42c33a00943690c6e4b888ee39206738783efe9ded"


@pytest.mark.parametrize("device", DEVICES)
def test_empty_bounded_and_normal_fills_keep_the_position(device):
    key = tt.Tandem(5).key
    for dtype in (torch.float32, torch.float64):
        assert tt.randn(key, 37, 0, dtype=dtype, device=device)[1] == 37
    for r in (3, 2**40):
        assert tt.randint(key, 37, 0, r, 0, device=device)[1] == 37


def test_randn_rounds_narrow_dtypes_from_float32():
    h = tt.Tandem(4).randn(100, dtype=torch.bfloat16)
    assert torch.equal(h, tt.Tandem(4).randn(100, dtype=torch.float32).to(torch.bfloat16))


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
        tt.Tandem(0).fill_(torch.empty(2, dtype=torch.float8_e4m3fn))


def test_entropy_seeds_differ():
    assert tt.Tandem().key != tt.Tandem().key


# ---- Bounded integers --------------------------------------------------------------------


def below(device, key, pos, r, n, wide):
    out = torch.empty(n, dtype=torch.uint64 if wide else torch.uint32, device=device)
    nxt = (tt._ext.fill_below_cuda if device == "cuda" else tt._ext.fill_below_cpu)(out, key, pos, 32, r)
    return out.cpu().tolist(), nxt


@pytest.mark.parametrize("device", DEVICES)
def test_bounded_fills_match_the_other_ports(device):
    """Rows of tandem-c and tandem-cuda, from tools/cross_json.py. The 2^31 + 1 rows reject
    often, so the fallback stream is exercised."""
    key = tt.Tandem(42).key
    assert tuple(CROSS["cuda_key"]) == key
    for wide, rows in ((False, CROSS["c_fill_below32"]), (True, CROSS["c_fill_below64"])):
        for row in rows:
            got, nxt = below(device, key, row["start"], row["range"], 64, wide)
            assert got == row["out"], (wide, row["range"])
            assert nxt == row["end_pos"]
    for wide, rows in ((False, CROSS["cuda_below32"]), (True, CROSS["cuda_below64"])):
        for row in rows:
            got, nxt = below(device, key, 0, row["range"], 64, wide)
            assert got == row["out"], (wide, row["range"])
            assert nxt == 64 * (64 if wide else 32)


@pytest.mark.parametrize("device", DEVICES)
def test_randint_ranges_and_widths(device):
    key = tt.Tandem(42).key
    got, nxt = tt.randint(key, 0, -5, 5, 40, device=device)
    want, _ = below(device, key, 0, 10, 40, False)
    assert got.dtype == torch.int64 and got.cpu().tolist() == [x - 5 for x in want] and nxt == 40 * 32
    # Above 2^32 the 64-bit fill draws, with the offset applied in wrapping arithmetic.
    lo, hi = 2**63, 2**63 + 2**40
    got, nxt = tt.randint(key, 0, lo, hi, (4, 5), dtype=torch.uint64, device=device)
    want, _ = below(device, key, 0, 2**40, 20, True)
    assert got.shape == (4, 5)
    assert [x % 2**64 for x in got.flatten().cpu().view(torch.int64).tolist()] == [lo + x for x in want]
    assert nxt == 20 * 64
    # The full 32-bit and 64-bit ranges are the raw words plus the offset.
    got, _ = tt.randint(key, 0, 0, 2**32, 9, dtype=torch.uint32, device=device)
    assert torch.equal(got.cpu(), tt.bits(key, 0, 9, dtype=torch.uint32)[0])
    got, _ = tt.randint(key, 0, -2**63, 2**63, 9, device=device)
    assert torch.equal(got.cpu(), tt.bits(key, 0, 9, dtype=torch.int64)[0] + (-2**63))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("r", [2**31 + 1, 2**63 + 1])
def test_randint_cut_at_any_boundary_equals_the_whole_call(device, r):
    """The fallback of a rejected draw is keyed by the global draw index, so chunks of a fill
    equal the fill. These ranges reject about half of the draws."""
    key, n = tt.Tandem(77).key, 300
    whole, end = tt.randint(key, 12345, 0, r, n, dtype=torch.uint64, device=device)
    for cut in (1, 77, 151, 299):
        a, mid = tt.randint(key, 12345, 0, r, cut, dtype=torch.uint64, device=device)
        b, end2 = tt.randint(key, mid, 0, r, n - cut, dtype=torch.uint64, device=device)
        assert torch.equal(torch.cat([a, b]).cpu(), whole.cpu()) and end2 == end, cut


def test_randint_dtype_bounds():
    t = tt.Tandem(1)
    x = t.randint(-128, 128, 1000, dtype=torch.int8)
    assert x.dtype == torch.int8 and x.min() < -100 and x.max() > 100
    with pytest.raises(ValueError):
        t.randint(0, 256, 3, dtype=torch.int8)
    with pytest.raises(ValueError):
        t.randint(4, 4, 3)


@pytest.mark.parametrize("dtype,low,high", [
    (torch.int32, -9, 1000), (torch.int32, -2**31, 2**31 - 1), (torch.uint32, 2**31 - 5, 2**32 - 1),
    (torch.int16, -300, 300), (torch.uint64, 2**64 - 2**33, 2**64 - 1)])
def test_randint_dtypes_are_the_shifted_draws(dtype, low, high):
    """The in-place paths for 32-bit and 64-bit results and the converting paths give the
    same integers as the draws below high - low plus low."""
    key = tt.Tandem(8).key
    got, nxt = tt.randint(key, 0, low, high, 1000, dtype=dtype)
    d, nxt0 = below("cpu", key, 0, high - low, 1000, high - low > 2**32)
    assert got.tolist() == [x + low for x in d] and nxt == nxt0


@pytest.mark.parametrize("device", DEVICES)
def test_out_gives_the_same_values_as_a_new_tensor(device):
    """out= fills in place for the draw's width and converts otherwise, also for strided out."""
    for dtype, lo, hi in ((torch.int32, -5, 50), (torch.int64, 0, 2**40), (torch.int64, -3, 9),
                          (torch.int16, -300, 300), (torch.uint8, 0, 256)):
        want, nxt = tt.randint(tt.Tandem(3).key, 0, lo, hi, (6, 8), dtype=dtype, device=device)
        for out in (torch.empty(6, 8, dtype=dtype, device=device),
                    torch.empty(6, 16, dtype=dtype, device=device)[:, ::2]):
            got, n2 = tt.randint(tt.Tandem(3).key, 0, lo, hi, out=out)
            assert got is out and n2 == nxt and torch.equal(out.cpu(), want.cpu()), (dtype, lo, hi)
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        want, nxt = tt.randn(tt.Tandem(3).key, 0, 6, 8, dtype=dtype, device=device)
        for out in (torch.empty(6, 8, dtype=dtype, device=device),
                    torch.empty(6, 16, dtype=dtype, device=device)[:, ::2]):
            got, n2 = tt.randn(tt.Tandem(3).key, 0, dtype=dtype, device=device, out=out)
            assert got is out and n2 == nxt and torch.equal(out.cpu(), want.cpu()), dtype


def lemire(words, n):
    """One scalar draw below n over an iterator of 32-bit stream words, as tandem_u32_below."""
    while True:
        m = next(words) * n
        if m & 0xffffffff >= (2**32 - n) % n:
            return m >> 32


def test_randperm_is_fisher_yates_over_scalar_draws():
    n = 200
    used = 0

    def stream():
        nonlocal used
        for w in tt.Tandem(11).bits(100_000, dtype=torch.uint32).tolist():
            used += 1
            yield w

    words = stream()
    want = list(range(n))
    for i in range(n - 1, 0, -1):
        j = lemire(words, i + 1)
        want[i], want[j] = want[j], want[i]
    t = tt.Tandem(11)
    assert t.randperm(n).tolist() == want
    assert t.position == 32 * used


@pytest.mark.parametrize("device", DEVICES)
def test_shuffle_permutes_along_dim(device):
    x = torch.arange(24, device=device).view(4, 6)
    y = tt.Tandem(2).shuffle(x, dim=1)
    perm = tt.Tandem(2).randperm(6)
    assert torch.equal(y.cpu(), x.cpu()[:, perm])


# ---- CUDA agrees with the CPU ------------------------------------------------------------

ALL_DTYPES = [torch.bool, torch.uint8, torch.int8, torch.uint16, torch.int16, torch.float16,
              torch.bfloat16, torch.uint32, torch.int32, torch.float32, torch.uint64, torch.int64,
              torch.float64, torch.complex64, torch.complex128]
UNIFORM = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64,
           torch.complex128)


@CUDA
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("K", [1, 8, 32, 128])
def test_cuda_equals_cpu(dtype, K):
    for pos in (0, 1, 8, 16, 31, 33, 64, 127, 128, 1000, 1024, 1025, 4096 * 7 + 3, 2**20 + 5):
        for n in (0, 1, 3, 16, 31, 32, 33, 1000, 2**16 + 7):
            key = (0xdeadbeef, pos & 0xffffffff, K, n)
            draw = tt.rand if dtype in UNIFORM else tt.bits
            a, na = draw(key, pos, n, dtype=dtype, K=K)
            b, nb = draw(key, pos, n, dtype=dtype, device="cuda", K=K)
            assert na == nb
            assert torch.equal(a, b.cpu()), (dtype, K, pos, n)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_normals_equal_cpu(dtype):
    for pos in (0, 33, 1000):
        for n in (0, 1, 7, 1001, 2**16 + 3):
            a, na = tt.randn((0xabc, pos, n, 1), pos, n, dtype=dtype)
            b, nb = tt.randn((0xabc, pos, n, 1), pos, n, dtype=dtype, device="cuda")
            assert na == nb and same_normals(b, a.tolist()), (dtype, pos, n)


@CUDA
@pytest.mark.parametrize("r", [3, 2**32 - 1, 2**40])
def test_cuda_randint_equals_cpu(r):
    for pos in (0, 33, 1000):
        for n in (0, 1, 7, 1001, 2**16 + 3):
            a, na = tt.randint((0xabc, pos, n, 1), pos, 0, r, n)
            b, nb = tt.randint((0xabc, pos, n, 1), pos, 0, r, n, device="cuda")
            assert na == nb and torch.equal(a, b.cpu()), (r, pos, n)


@CUDA
@pytest.mark.parametrize("dtype,lo,hi", [
    (torch.int32, -7, 1000), (torch.uint32, 5, 2**31 + 1), (torch.int64, -2**40, 1000),
    (torch.uint64, 2**63, 2**63 + 2**40), (torch.int64, -2**62, 2**62), (torch.int8, -3, 100)])
def test_cuda_randint_fused_low_and_widening_equal_cpu(dtype, lo, hi):
    key = tt.Tandem(9).key
    a, na = tt.randint(key, 33, lo, hi, 1001, dtype=dtype)
    b, nb = tt.randint(key, 33, lo, hi, 1001, dtype=dtype, device="cuda")
    out = torch.empty(1001, dtype=dtype, device="cuda")
    c, nc = tt.randint(key, 33, lo, hi, out=out)
    assert na == nb == nc and torch.equal(a, b.cpu()) and c is out and torch.equal(a, out.cpu())


@CUDA
def test_cuda_unaligned_storage():
    big = torch.empty(2**16 + 9, dtype=torch.uint32, device="cuda")
    view = big[1:]  # 4 bytes off a 16-byte boundary
    tt.Tandem(7).fill_(view)
    assert torch.equal(view.cpu(), tt.Tandem(7).bits(2**16 + 8, dtype=torch.uint32))
