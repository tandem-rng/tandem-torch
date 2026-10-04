// Tensor fills and key derivations over the C reference. The CUDA fills live in _cuda.cu.
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "tandem.h"
}

namespace {

using Key = std::array<uint32_t, 4>;

tandem_rng make(const Key &key, uint64_t pos, uint32_t K) {
    return tandem_from_key(key.data(), pos, K);
}

Key key_of(const tandem_rng &rng) {
    Key k;
    tandem_key(&rng, k.data());
    return k;
}

/* Fill a contiguous CPU tensor in place from an aligned-by-the-callee position. Returns the
 * position after the fill. Signed integers and the half type take the unsigned bytes. */
uint64_t fill_cpu(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cpu(), "fill_cpu: tensor must be on the CPU");
    TORCH_CHECK(out.is_contiguous(), "fill_cpu: tensor must be contiguous");
    tandem_rng rng = make(key, pos, K);
    size_t n = (size_t)out.numel();
    void *p = out.data_ptr();
    switch (out.scalar_type()) {
    case torch::kBool: tandem_fill_bool(&rng, static_cast<bool *>(p), n); break;
    case torch::kUInt8:
    case torch::kInt8: tandem_fill_u8(&rng, static_cast<uint8_t *>(p), n); break;
    case torch::kUInt16:
    case torch::kInt16: tandem_fill_u16(&rng, static_cast<uint16_t *>(p), n); break;
    case torch::kHalf: tandem_fill_f16_bits(&rng, static_cast<uint16_t *>(p), n); break;
    case torch::kUInt32:
    case torch::kInt32: tandem_fill_u32(&rng, static_cast<uint32_t *>(p), n); break;
    case torch::kFloat: tandem_fill_f32(&rng, static_cast<float *>(p), n); break;
    case torch::kUInt64:
    case torch::kInt64: tandem_fill_u64(&rng, static_cast<uint64_t *>(p), n); break;
    case torch::kDouble: tandem_fill_f64(&rng, static_cast<double *>(p), n); break;
    case torch::kComplexFloat: tandem_fill_c32(&rng, static_cast<float *>(p), n); break;
    case torch::kComplexDouble: tandem_fill_c64(&rng, static_cast<double *>(p), n); break;
    default: TORCH_CHECK(false, "fill_cpu: unsupported dtype ", out.scalar_type());
    }
    return tandem_position(&rng);
}

/* Bounded fill of a uint32 or uint64 tensor: element i is draw i of the fill below `range`.
 * Rejected draws retry on a fallback stream, so the values equal the CUDA fill. */
uint64_t fill_below_cpu(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K, uint64_t range) {
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous(), "fill_below_cpu: need a contiguous CPU tensor");
    tandem_rng rng = make(key, pos, K);
    size_t n = (size_t)out.numel();
    switch (out.scalar_type()) {
    case torch::kUInt32:
        TORCH_CHECK(range <= UINT32_MAX, "fill_below_cpu: range does not fit 32 bits");
        tandem_fill_u32_below(&rng, static_cast<uint32_t *>(out.data_ptr()), n, (uint32_t)range);
        break;
    case torch::kUInt64: tandem_fill_u64_below(&rng, static_cast<uint64_t *>(out.data_ptr()), n, range); break;
    default: TORCH_CHECK(false, "fill_below_cpu: dtype must be uint32 or uint64");
    }
    return tandem_position(&rng);
}

/* Standard normals into a float32 or float64 tensor, the Box-Muller fills of tandem-c. */
uint64_t fill_normal_cpu(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous(), "fill_normal_cpu: need a contiguous CPU tensor");
    tandem_rng rng = make(key, pos, K);
    size_t n = (size_t)out.numel();
    switch (out.scalar_type()) {
    case torch::kFloat: tandem_fill_normal_f32(&rng, out.data_ptr<float>(), n); break;
    case torch::kDouble: tandem_fill_normal_f64(&rng, out.data_ptr<double>(), n); break;
    default: TORCH_CHECK(false, "fill_normal_cpu: dtype must be float32 or float64");
    }
    return tandem_position(&rng);
}

/* Standard exponentials into a float32 or float64 tensor, -log(1 - u) of the plain float fill. */
uint64_t fill_exponential_cpu(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous(), "fill_exponential_cpu: need a contiguous CPU tensor");
    tandem_rng rng = make(key, pos, K);
    size_t n = (size_t)out.numel();
    switch (out.scalar_type()) {
    case torch::kFloat: tandem_fill_exponential_f32(&rng, out.data_ptr<float>(), n); break;
    case torch::kDouble: tandem_fill_exponential_f64(&rng, out.data_ptr<double>(), n); break;
    default: TORCH_CHECK(false, "fill_exponential_cpu: dtype must be float32 or float64");
    }
    return tandem_position(&rng);
}

/* Fisher-Yates from the end with one sequential scalar bounded draw per step. The scalar draw
 * rejects by discarding, so the position after the shuffle depends on the draws. */
std::pair<torch::Tensor, uint64_t> randperm_cpu(int64_t n, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(n >= 0 && n <= (int64_t)UINT32_MAX, "randperm_cpu: n must be in [0, 2^32 - 1]");
    tandem_rng rng = make(key, pos, K);
    torch::Tensor out = torch::empty({n}, torch::kInt64);
    int64_t *p = out.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; i++) p[i] = i;
    for (int64_t i = n - 1; i > 0; i--) {
        int64_t j = tandem_u32_below(&rng, (uint32_t)(i + 1));
        std::swap(p[i], p[j]);
    }
    return {out, tandem_position(&rng)};
}

/* Element i of the fill that would start at `pos`, as a Python number. */
py::object at_value(const Key &key, uint64_t pos, uint32_t K, const std::string &kind, uint64_t i) {
    tandem_rng rng = make(key, pos, K);
    if (kind == "u32") return py::int_(tandem_at_u32(&rng, i));
    if (kind == "u64") return py::int_(tandem_at_u64(&rng, i));
    if (kind == "f32") return py::float_(tandem_at_f32(&rng, i));
    if (kind == "f64") return py::float_(tandem_at_f64(&rng, i));
    TORCH_CHECK(false, "at: unknown kind ", kind);
}

Key seed_key(uint64_t lo, uint64_t hi) {
    return key_of(tandem_seed(lo, hi, 0));
}

Key split_key(const Key &key, uint64_t index) {
    tandem_rng rng = make(key, 0, 0);
    return key_of(tandem_split(&rng, index));
}

Key sub_key(const Key &key, uint64_t purpose) {
    tandem_rng rng = make(key, 0, 0);
    return key_of(tandem_sub(&rng, purpose));
}

/* The children of a fork at `pos` and the parent's position after it. */
std::pair<std::vector<Key>, uint64_t> fork_keys(const Key &key, uint64_t pos, uint64_t n) {
    tandem_rng rng = make(key, pos, 0);
    std::vector<tandem_rng> kids(n);
    tandem_fork(&rng, kids.data(), n);
    std::vector<Key> keys;
    keys.reserve(n);
    for (const tandem_rng &k : kids) keys.push_back(key_of(k));
    return {keys, tandem_position(&rng)};
}

} // namespace

#ifdef TANDEM_TORCH_CUDA
uint64_t fill_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K);
uint64_t fill_below_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K, uint64_t range);
uint64_t randint_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K, uint64_t range, int64_t low);
uint64_t fill_normal_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K);
uint64_t fill_exponential_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_cpu", &fill_cpu);
    m.def("fill_below_cpu", &fill_below_cpu);
    m.def("fill_normal_cpu", &fill_normal_cpu);
    m.def("fill_exponential_cpu", &fill_exponential_cpu);
    m.def("randperm_cpu", &randperm_cpu);
    m.def("at", &at_value);
    m.def("seed", &seed_key);
    m.def("split", &split_key);
    m.def("sub", &sub_key);
    m.def("fork", &fork_keys);
#ifdef TANDEM_TORCH_CUDA
    m.def("fill_cuda", &fill_cuda);
    m.def("fill_below_cuda", &fill_below_cuda);
    m.def("randint_cuda", &randint_cuda);
    m.def("fill_normal_cuda", &fill_normal_cuda);
    m.def("fill_exponential_cuda", &fill_exponential_cuda);
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif
}
