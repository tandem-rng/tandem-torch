"""Throughput of 2**27-element fills against torch's own generator, into preallocated tensors, and
on CUDA against cuRAND Philox4x32-10 for each output type.

CPU: minimum of 7 wall-clock timings after a warm-up. CUDA: cudaEvent timings, half a second of
warm-up, minimum of 21. Usage: python tools/bench.py [cpu|cuda]
"""

import sys
import time

import torch

import tandem_torch as tt

N = 2**27
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
t = tt.Tandem(42)
gen = torch.Generator(device=device)
gen.manual_seed(42)


def gibs(fn, nbytes):
    if device == "cuda":
        fn()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(21):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            best = min(best, a.elapsed_time(b) / 1e3)
    else:
        fn()
        best = float("inf")
        for _ in range(7):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
    return nbytes / best / 2**30


rows = []
for dtype in (torch.float32, torch.float64):
    buf = torch.empty(N, dtype=dtype, device=device)
    name = str(dtype).removeprefix("torch.")
    rows.append((f"Tandem.rand {name}", gibs(lambda: t.rand(out=buf), buf.nbytes)))
    rows.append((f"torch.rand {name}",
                 gibs(lambda: torch.rand(N, dtype=dtype, device=device, generator=gen, out=buf), buf.nbytes)))
u32 = torch.empty(N, dtype=torch.uint32, device=device)
rows.append(("Tandem.bits uint32", gibs(lambda: t.bits(out=u32), u32.nbytes)))

# The narrow types, which CUDA fills in their own kernels.
for dtype in (torch.uint8, torch.bool, torch.float16):
    buf = torch.empty(N, dtype=dtype, device=device)
    name = str(dtype).removeprefix("torch.")
    fill = (lambda: t.rand(out=buf)) if dtype == torch.float16 else (lambda: t.bits(out=buf))
    rows.append((f"Tandem fill {name}", gibs(fill, buf.nbytes)))

# Tandem.randint and Tandem.randn with out= write in place, without out= they allocate. The
# torch calls write into a preallocated tensor.
for dtype, lo, hi, label in ((torch.int32, -2**31, 2**31 - 1, "full range"), (torch.int32, 0, 1000, "range 1000"),
                             (torch.int64, 0, 1000, "range 1000"), (torch.int64, -2**62, 2**62, "range 2^63")):
    buf = torch.empty(N, dtype=dtype, device=device)
    name = str(dtype).removeprefix("torch.")
    rows.append((f"Tandem.randint {name} {label} out=", gibs(lambda: t.randint(lo, hi, out=buf), buf.nbytes)))
    rows.append((f"Tandem.randint {name} {label}", gibs(lambda: t.randint(lo, hi, N, dtype=dtype, device=device), buf.nbytes)))
    rows.append((f"torch.randint {name} {label}", gibs(
        lambda: torch.randint(lo, hi, (N,), dtype=dtype, device=device, generator=gen, out=buf), buf.nbytes)))
for dtype in (torch.float32, torch.float64):
    buf = torch.empty(N, dtype=dtype, device=device)
    name = str(dtype).removeprefix("torch.")
    rows.append((f"Tandem.randn {name} out=", gibs(lambda: t.randn(out=buf), buf.nbytes)))
    rows.append((f"Tandem.randn {name}", gibs(lambda: t.randn(N, dtype=dtype, device=device), buf.nbytes)))
    rows.append((f"torch.randn {name}",
                 gibs(lambda: torch.randn(N, dtype=dtype, device=device, generator=gen, out=buf), buf.nbytes)))
for dtype in (torch.float32, torch.float64):
    buf = torch.empty(N, dtype=dtype, device=device)
    name = str(dtype).removeprefix("torch.")
    rows.append((f"Tandem.exponential {name} out=", gibs(lambda: t.exponential(out=buf), buf.nbytes)))
    rows.append((f"Tensor.exponential_ {name}", gibs(lambda: buf.exponential_(generator=gen), buf.nbytes)))

if device == "cuda":
    # cuRAND's host API on torch's stream, through the libcurand that PyTorch's CUDA wheel installs.
    # cuRAND has no 8-, 16- or 64-bit integer output for Philox, no bounded integers and no
    # exponentials: those rows take the nearest call, 32-bit words into the same bytes or the
    # uniform the exponential reads.
    import ctypes
    import os

    import nvidia.curand

    cr = ctypes.CDLL(os.path.join(nvidia.curand.__path__[0], "lib", "libcurand.so.10"))
    cg = ctypes.c_void_p()
    assert cr.curandCreateGenerator(ctypes.byref(cg), 161) == 0  # CURAND_RNG_PSEUDO_PHILOX4_32_10
    assert cr.curandSetPseudoRandomGeneratorSeed(cg, ctypes.c_ulonglong(42)) == 0
    assert cr.curandSetStream(cg, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)) == 0

    def curand(call, buf, count, *args):
        out, n = ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(count)
        return lambda: getattr(cr, call)(cg, out, n, *args)

    normal = {torch.float32: (ctypes.c_float(0), ctypes.c_float(1)),
              torch.float64: (ctypes.c_double(0), ctypes.c_double(1))}
    for call, dtype, words, extra in (
            ("curandGenerateUniform", torch.float32, 1, ()), ("curandGenerateUniformDouble", torch.float64, 1, ()),
            ("curandGenerate", torch.uint32, 1, ()), ("curandGenerate", torch.uint8, 0.25, ()),
            ("curandGenerate", torch.int64, 2, ()), ("curandGenerateNormal", torch.float32, 1, normal[torch.float32]),
            ("curandGenerateNormalDouble", torch.float64, 1, normal[torch.float64])):
        buf = torch.empty(N, dtype=dtype, device=device)
        name = str(dtype).removeprefix("torch.")
        rows.append((f"cuRAND {call} {name}", gibs(curand(call, buf, int(N * words), *extra), buf.nbytes)))

for name, g in rows:
    print(f"{name:44s} {g:8.1f} GiB/s")
print("device", device, torch.cuda.get_device_name() if device == "cuda" else "")
