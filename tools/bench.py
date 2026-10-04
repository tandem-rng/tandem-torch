"""Throughput of 2**27-element fills against torch's own generator, into preallocated tensors.

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
i32 = torch.empty(N, dtype=torch.int32, device=device)
rows.append(("Tandem.bits uint32", gibs(lambda: t.bits(out=u32), u32.nbytes)))
rows.append(("torch.randint int32", gibs(
    lambda: torch.randint(-2**31, 2**31 - 1, (N,), dtype=torch.int32, device=device, generator=gen, out=i32),
    i32.nbytes)))

# The narrow types, which CUDA fills in their own kernels.
for dtype in (torch.uint8, torch.bool, torch.float16):
    buf = torch.empty(N, dtype=dtype, device=device)
    name = str(dtype).removeprefix("torch.")
    fill = (lambda: t.rand(out=buf)) if dtype == torch.float16 else (lambda: t.bits(out=buf))
    rows.append((f"Tandem fill {name}", gibs(fill, buf.nbytes)))

# randint and randn allocate their result, the torch calls write into a preallocated tensor.
rows.append(("Tandem.randint int32", gibs(lambda: t.randint(-2**31, 2**31 - 1, N, dtype=torch.int32, device=device),
                                          4 * N)))
for dtype in (torch.float32, torch.float64):
    buf = torch.empty(N, dtype=dtype, device=device)
    name = str(dtype).removeprefix("torch.")
    rows.append((f"Tandem.randn {name}", gibs(lambda: t.randn(N, dtype=dtype, device=device), buf.nbytes)))
    rows.append((f"torch.randn {name}",
                 gibs(lambda: torch.randn(N, dtype=dtype, device=device, generator=gen, out=buf), buf.nbytes)))

for name, g in rows:
    print(f"{name:28s} {g:8.1f} GiB/s")
print("device", device, torch.cuda.get_device_name() if device == "cuda" else "")
