// Tensor fills and key derivations over the C reference. The CUDA fills live in _cuda.cu.
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <stdexcept>

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
    default: TORCH_CHECK(false, "fill_cpu: unsupported dtype ", out.scalar_type());
    }
    return tandem_position(&rng);
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
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_cpu", &fill_cpu);
    m.def("seed", &seed_key);
    m.def("split", &split_key);
    m.def("sub", &sub_key);
    m.def("fork", &fork_keys);
#ifdef TANDEM_TORCH_CUDA
    m.def("fill_cuda", &fill_cuda);
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif
}
