"""PyTorch tensors from Tandem8x32.

A :class:`Tandem` holds the transport form of the specification (128-bit key, chunk length
``K``) and a stream bit position. Every fill aligns the position to the element width, reads,
and advances past the elements, so the tensors equal the fills of the other implementations
for the same key and position. The CPU path is the reference C implementation, the CUDA path the reference CUDA header.
"""

import math
import os

import torch

from . import _ext

__all__ = ["Tandem", "rand", "bits", "randint", "randperm", "randn", "has_cuda"]

has_cuda = _ext.has_cuda

_BITS = {
    torch.bool: 1,
    torch.uint8: 8, torch.int8: 8,
    torch.uint16: 16, torch.int16: 16, torch.float16: 16,
    torch.uint32: 32, torch.int32: 32, torch.float32: 32,
    torch.uint64: 64, torch.int64: 64, torch.float64: 64,
    torch.bfloat16: 16, torch.complex64: 64, torch.complex128: 128,
}
_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_COMPLEX = (torch.complex64, torch.complex128)
_INTS = tuple(d for d in _BITS if d not in _FLOATS + _COMPLEX + (torch.bool,))
_U = {8: torch.uint8, 16: torch.uint16, 32: torch.uint32, 64: torch.uint64}


def _align(pos, w):
    return (pos + w - 1) & ~(w - 1)


def _check_key(key):
    key = tuple(int(k) for k in key)
    if len(key) != 4 or any(not 0 <= k < 2**32 for k in key):
        raise ValueError("key must be four unsigned 32-bit words")
    return key


def _fill(out, key, pos, K):
    """Fill a contiguous tensor in place. Returns the position after the fill."""
    w = _BITS.get(out.dtype)
    if w is None:
        raise TypeError(f"unsupported dtype {out.dtype}")
    if not out.is_contiguous():
        tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        nxt = _fill(tmp, key, pos, K)
        out.copy_(tmp)
        return nxt
    if out.dtype == torch.bfloat16:
        # Not in the specification: the float16 rule with 8 fraction bits, (raw16 >> 8) * 2^-8,
        # which a bfloat16 holds exactly.
        nxt = _fill(out.view(torch.uint16), key, pos, K)
        k = out.view(torch.uint16).to(torch.int32) >> 8
        out.copy_(k.to(torch.float32) * 2.0**-8)
        return nxt
    if out.device.type == "cpu":
        return _ext.fill_cpu(out, key, pos, K)
    if out.device.type != "cuda":
        raise RuntimeError(f"tandem_torch fills CPU and CUDA tensors, not {out.device.type}")
    if not has_cuda:
        raise RuntimeError("tandem_torch was built without CUDA support")
    if w >= 32:
        return _ext.fill_cuda(out, key, pos, K)
    return _fill_cuda_narrow(out, key, pos, K, w)


def _fill_cuda_narrow(out, key, pos, K, w):
    """Elements narrower than a word come from a word fill over the same stream bytes, since
    after alignment every fill is one byte stream."""
    n = out.numel()
    p = _align(pos, w)
    p32 = p & ~31
    nbits = n * w
    nwords = -(-(p - p32 + nbits) // 32)
    words = torch.empty(nwords, dtype=torch.uint32, device=out.device)
    _ext.fill_cuda(words, key, p32, K)
    raw = words.view(torch.uint8)
    if w == 1:
        off = p - p32
        bit = (raw.unsqueeze(1) >> torch.arange(8, dtype=torch.uint8, device=out.device)) & 1
        out.view(-1).copy_(bit.flatten()[off:off + n])
    else:
        off = (p - p32) // 8
        raw = raw[off:off + nbits // 8]
        if out.dtype == torch.float16:
            # (raw >> 5) * 2^-11 has at most 11 significant bits, so the half is exact.
            k = raw.view(torch.uint16).to(torch.int32) >> 5
            out.view(-1).copy_(k.to(torch.float32) * 2.0**-11)
        else:
            out.view(torch.uint8).copy_(raw)
    return p + nbits


def _target(shape, dtype, default, device, out, allowed):
    """The tensor to fill: `out`, whose dtype must fit the call, or a new one."""
    if out is not None:
        if shape and tuple(out.shape) != tuple(shape):
            raise ValueError("out has a different shape")
        if dtype is not None and out.dtype != dtype:
            raise TypeError("out has a different dtype")
        dtype = out.dtype
    dtype = default if dtype is None else dtype
    if not allowed(dtype):
        raise TypeError(f"unsupported dtype {dtype} for this call")
    return out if out is not None else torch.empty(shape, dtype=dtype, device=device)


def rand(key, position, *shape, dtype=None, device="cpu", K=32, out=None):
    """Uniform draws in [0, 1) with the specification's float mappings, from a key and position.
    The dtype defaults to float64. A complex element is its real then imaginary component.
    bfloat16 is a tandem-torch extension, ``(raw16 >> 8) * 2^-8``. Returns
    ``(tensor, next_position)``."""
    t = _target(shape, dtype, torch.float64, device, out, lambda d: d in _FLOATS + _COMPLEX)
    return t, _fill(t, _check_key(key), position, K)


def bits(key, position, *shape, dtype=None, device="cpu", K=32, out=None):
    """Stream words as integers, uint64 by default. Signed types are the unsigned draws
    reinterpreted. Returns ``(tensor, next_position)``."""
    t = _target(shape, dtype, torch.uint64, device, out, lambda d: d in _BITS and d not in _FLOATS + _COMPLEX)
    return t, _fill(t, _check_key(key), position, K)


def _below(key, pos, K, r, n, device):
    """n draws below r, as uint32 (r <= 2^32) or uint64 (r <= 2^64) values. The raw word fill
    is the draw below 2^32 or 2^64, so those two ranges take no rejection."""
    wide = r > 2**32
    d = torch.empty(n, dtype=torch.uint64 if wide else torch.uint32, device=device)
    if r == 2**32 or r == 2**64:
        return d, _fill(d, key, pos, K)
    if torch.device(device).type == "cpu":
        return d, _ext.fill_below_cpu(d, key, pos, K, r)
    if not has_cuda:
        raise RuntimeError("tandem_torch was built without CUDA support")
    return d, _ext.fill_below_cuda(d, key, pos, K, r)


def randint(key, position, low, high, size, *, dtype=torch.int64, device="cpu", K=32):
    """Integers uniform on ``[low, high)`` by Lemire's method, as ``tandem::fill_u32_below``
    when the range is at most 2^32 and ``fill_u64_below`` above. Each element consumes one
    32-bit or 64-bit draw, a rejected draw retries on a fallback stream, and CPU and CUDA give
    the same values. Not part of the specification. Returns ``(tensor, next_position)``."""
    if dtype not in _INTS:
        raise TypeError(f"unsupported dtype {dtype} for randint")
    low, high = int(low), int(high)
    info = torch.iinfo(dtype)
    if not info.min <= low < high <= info.max + 1:
        raise ValueError(f"need {info.min} <= low < high <= {info.max + 1} for {dtype}")
    shape = (size,) if isinstance(size, int) else tuple(size)
    d, nxt = _below(_check_key(key), position, K, high - low, math.prod(shape), device)
    # A draw added to low in wrapping arithmetic of the draw's width is exact once the result
    # fits dtype. The words of an int32 or int64 result are filled in place, others convert.
    if d.dtype == torch.uint32 and dtype in (torch.int32, torch.uint32):
        t, bits = d.view(torch.int32), 32
    elif d.dtype == torch.uint64 and dtype in (torch.int64, torch.uint64):
        t, bits = d.view(torch.int64), 64
    else:
        t, bits = d.to(torch.int64), 64
    if low:
        t.add_(((low + 2**(bits - 1)) % 2**bits) - 2**(bits - 1))
    t = t.view(dtype) if info.bits == bits else t.to(dtype)
    return t.view(shape), nxt


def randn(key, position, *shape, dtype=torch.float64, device="cpu", K=32):
    """Standard normals by Box-Muller, as ``tandem_fill_normal_f64`` and ``_f32`` on CPU and
    ``tandem::fill_normal_f64`` and ``_f32`` on CUDA. A float64 normal is made from two float64
    uniforms (128 stream bits), a float32 normal from two float32 uniforms (64 bits) in float
    arithmetic. The libm and CUDA math functions differ in the last bits, so devices agree to a
    few ulps. Other float dtypes round the float32 normal. Not part of the specification.
    Returns ``(tensor, next_position)``."""
    if dtype not in _FLOATS:
        raise TypeError(f"unsupported dtype {dtype} for randn")
    native = dtype if dtype in (torch.float32, torch.float64) else torch.float32
    t = torch.empty(shape, dtype=native, device=device)
    key = _check_key(key)
    if t.numel() == 0:
        # tandem-c's empty normal fill leaves the position unaligned, tandem-cuda's and every
        # other fill align it. Align here so both devices and all dtypes agree.
        return t.to(dtype), _align(position, 8 * t.element_size())
    if t.device.type == "cpu":
        nxt = _ext.fill_normal_cpu(t, key, position, K)
    elif t.device.type == "cuda" and has_cuda:
        nxt = _ext.fill_normal_cuda(t, key, position, K)
    else:
        raise RuntimeError(f"tandem_torch fills CPU and CUDA tensors, not {t.device.type}")
    return t.to(dtype), nxt


def randperm(key, position, n, *, dtype=torch.int64, device="cpu", K=32):
    """A uniform permutation of ``range(n)`` by Fisher-Yates from the end: for i = n-1 down to
    1, swap element i with element j, where j is a scalar draw below i + 1 (Lemire, one draw
    after another, a rejected draw discarded). The CPU defines the permutation and a CUDA result
    is copied there. Not part of the specification. Returns ``(tensor, next_position)``."""
    if dtype not in _INTS:
        raise TypeError(f"unsupported dtype {dtype} for randperm")
    if n > torch.iinfo(dtype).max + 1:
        raise ValueError(f"n does not fit {dtype}")
    perm, nxt = _ext.randperm_cpu(int(n), _check_key(key), position, K)
    return perm.to(dtype=dtype, device=device), nxt


class Tandem:
    """A Tandem8x32 generator: key, chunk length and the stream position of the next draw.

    ``seed`` is an integer in ``[0, 2**128)`` and goes through the specification's seed
    whitening, so ``Tandem(42)`` produces the stream of Julia ``Tandem8x32(42)`` and C
    ``tandem_seed(42, 0, K)``. ``None`` draws 128 bits of OS entropy.
    """

    __slots__ = ("key", "position", "chunk_length")

    def __init__(self, seed=None, K=32):
        if seed is None:
            seed = int.from_bytes(os.urandom(16), "little")
        seed = int(seed)
        if not 0 <= seed < 2**128:
            raise ValueError("seed must be in [0, 2**128)")
        self.key = tuple(_ext.seed(seed & (2**64 - 1), seed >> 64))
        self.position = 0
        self.chunk_length = _check_K(K)

    @classmethod
    def from_key(cls, key, position=0, K=32):
        self = cls.__new__(cls)
        self.key = _check_key(key)
        self.position = _check_position(position)
        self.chunk_length = _check_K(K)
        return self

    def __repr__(self):
        return f"Tandem(key={self.key}, position={self.position}, K={self.chunk_length})"

    def __eq__(self, other):
        return isinstance(other, Tandem) and (self.key, self.position, self.chunk_length) == (
            other.key, other.position, other.chunk_length)

    def __hash__(self):
        return hash((self.key, self.position, self.chunk_length))

    # Draws --------------------------------------------------------------------------------

    def fill_(self, tensor):
        """Fill a tensor in place with the next draws of its dtype and advance."""
        self.position = _fill(tensor, self.key, self.position, self.chunk_length)
        return tensor

    def rand(self, *shape, dtype=None, device="cpu", out=None):
        """Uniform draws in [0, 1): 53 random bits for float64, 24 for float32, 11 for float16,
        8 for bfloat16. Complex dtypes draw the real then the imaginary component."""
        t, self.position = rand(self.key, self.position, *shape, dtype=dtype, device=device,
                                K=self.chunk_length, out=out)
        return t

    def bits(self, *shape, dtype=None, device="cpu", out=None):
        """Stream words as integers."""
        t, self.position = bits(self.key, self.position, *shape, dtype=dtype, device=device,
                                K=self.chunk_length, out=out)
        return t

    def randbool(self, *shape, device="cpu", out=None):
        """One stream bit per element."""
        return self.bits(*shape, dtype=torch.bool, device=device, out=out)

    def randint(self, low, high, size, *, dtype=torch.int64, device="cpu"):
        """Integers uniform on ``[low, high)``, see :func:`randint`."""
        t, self.position = randint(self.key, self.position, low, high, size, dtype=dtype,
                                   device=device, K=self.chunk_length)
        return t

    def randperm(self, n, *, dtype=torch.int64, device="cpu"):
        """A uniform permutation of ``range(n)``, see :func:`randperm`."""
        t, self.position = randperm(self.key, self.position, n, dtype=dtype, device=device,
                                    K=self.chunk_length)
        return t

    def shuffle(self, x, dim=0):
        """``x`` with its entries along ``dim`` in a random order, as a new tensor."""
        return x.index_select(dim, self.randperm(x.shape[dim], device=x.device))

    def randn(self, *shape, dtype=torch.float64, device="cpu"):
        """Standard normal draws by Box-Muller, see :func:`randn`."""
        t, self.position = randn(self.key, self.position, *shape, dtype=dtype, device=device,
                                 K=self.chunk_length)
        return t

    # Derived generators --------------------------------------------------------------------

    def split(self, index):
        """Child ``index`` by key alone, at position 0 with the same K."""
        return Tandem.from_key(_ext.split(self.key, _check_u64(index)), 0, self.chunk_length)

    def sub(self, purpose):
        """A generator for a named purpose, by key alone."""
        return Tandem.from_key(_ext.sub(self.key, _check_u64(purpose)), 0, self.chunk_length)

    def fork(self, n):
        """``n`` children from the current block. The parent moves past the block."""
        keys, self.position = _ext.fork(self.key, self.position, _check_u64(n))
        return [Tandem.from_key(k, 0, self.chunk_length) for k in keys]


def _check_K(K):
    K = int(K)
    if K < 1 or K > 65536 or K & (K - 1):
        raise ValueError("K must be a power of two in [1, 65536]")
    return K


def _check_position(position):
    position = int(position)
    if not 0 <= position < 2**63:
        raise ValueError("position must be in [0, 2**63)")
    return position


def _check_u64(x):
    x = int(x)
    if not 0 <= x < 2**64:
        raise ValueError("value must be in [0, 2**64)")
    return x
