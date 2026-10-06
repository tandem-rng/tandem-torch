// CUDA fills over the vendored CUDA header, on the tensor's current stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "tandem.cuh"

using Key = std::array<uint32_t, 4>;

uint64_t fill_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cuda(), "fill_cuda: tensor must be on a CUDA device");
    TORCH_CHECK(out.is_contiguous(), "fill_cuda: tensor must be contiguous");
    c10::cuda::CUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t n = (size_t)out.numel();
    void *p = out.data_ptr();
    uint64_t next;
    switch (out.scalar_type()) {
    case torch::kBool: next = tandem::fill_bool(key.data(), pos, K, static_cast<bool *>(p), n, stream); break;
    case torch::kUInt8:
    case torch::kInt8: next = tandem::fill_u8(key.data(), pos, K, static_cast<uint8_t *>(p), n, stream); break;
    case torch::kUInt16:
    case torch::kInt16: next = tandem::fill_u16(key.data(), pos, K, static_cast<uint16_t *>(p), n, stream); break;
    case torch::kHalf: next = tandem::fill_f16_bits(key.data(), pos, K, static_cast<uint16_t *>(p), n, stream); break;
    case torch::kUInt32:
    case torch::kInt32: next = tandem::fill_u32(key.data(), pos, K, static_cast<uint32_t *>(p), n, stream); break;
    case torch::kFloat: next = tandem::fill_f32(key.data(), pos, K, static_cast<float *>(p), n, stream); break;
    case torch::kUInt64:
    case torch::kInt64: next = tandem::fill_u64(key.data(), pos, K, static_cast<uint64_t *>(p), n, stream); break;
    case torch::kDouble: next = tandem::fill_f64(key.data(), pos, K, static_cast<double *>(p), n, stream); break;
    // A complex element is its real then imaginary component, so the fill is 2n components.
    case torch::kComplexFloat: next = tandem::fill_f32(key.data(), pos, K, static_cast<float *>(p), 2 * n, stream); break;
    case torch::kComplexDouble: next = tandem::fill_f64(key.data(), pos, K, static_cast<double *>(p), 2 * n, stream); break;
    default: TORCH_CHECK(false, "fill_cuda: unsupported dtype ", out.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next;
}

uint64_t fill_below_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K, uint64_t range) {
    TORCH_CHECK(out.device().is_cuda() && out.is_contiguous(), "fill_below_cuda: need a contiguous CUDA tensor");
    c10::cuda::CUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t n = (size_t)out.numel();
    uint64_t next;
    switch (out.scalar_type()) {
    case torch::kUInt32:
        TORCH_CHECK(range <= UINT32_MAX, "fill_below_cuda: range does not fit 32 bits");
        next = tandem::fill_u32_below(key.data(), pos, K, (uint32_t)range, static_cast<uint32_t *>(out.data_ptr()), n, stream);
        break;
    case torch::kUInt64:
        next = tandem::fill_u64_below(key.data(), pos, K, range, static_cast<uint64_t *>(out.data_ptr()), n, stream);
        break;
    default: TORCH_CHECK(false, "fill_below_cuda: dtype must be uint32 or uint64");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next;
}

// The low bound and the widening are fused into the store, so there is no second pass over out.
// A 32-bit range into a 32-bit or 64-bit element, a 64-bit range into a 64-bit element.
uint64_t randint_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K, uint64_t range, int64_t low) {
    TORCH_CHECK(out.device().is_cuda() && out.is_contiguous(), "randint_cuda: need a contiguous CUDA tensor");
    c10::cuda::CUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t n = (size_t)out.numel();
    void *p = out.data_ptr();
    const bool wide = range > UINT32_MAX;
    uint64_t next;
#define TANDEM_RANDINT(W, R, O) \
    next = tandem::fill_##W##_below(key.data(), pos, K, (R)range, (O)low, static_cast<O *>(p), n, stream)
    switch (out.scalar_type()) {
    case torch::kInt:
        TORCH_CHECK(!wide, "randint_cuda: range does not fit 32 bits");
        TANDEM_RANDINT(u32, uint32_t, int32_t);
        break;
    case torch::kUInt32:
        TORCH_CHECK(!wide, "randint_cuda: range does not fit 32 bits");
        TANDEM_RANDINT(u32, uint32_t, uint32_t);
        break;
    case torch::kLong:
        if (wide) TANDEM_RANDINT(u64, uint64_t, int64_t); else TANDEM_RANDINT(u32, uint32_t, int64_t);
        break;
    case torch::kUInt64:
        if (wide) TANDEM_RANDINT(u64, uint64_t, uint64_t); else TANDEM_RANDINT(u32, uint32_t, uint64_t);
        break;
    default: TORCH_CHECK(false, "randint_cuda: dtype must be int32, uint32, int64 or uint64");
    }
#undef TANDEM_RANDINT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next;
}

uint64_t fill_normal_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cuda() && out.is_contiguous(), "fill_normal_cuda: need a contiguous CUDA tensor");
    c10::cuda::CUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t n = (size_t)out.numel();
    uint64_t next;
    switch (out.scalar_type()) {
    case torch::kFloat: next = tandem::fill_normal_f32(key.data(), pos, K, out.data_ptr<float>(), n, stream); break;
    case torch::kDouble: next = tandem::fill_normal_f64(key.data(), pos, K, out.data_ptr<double>(), n, stream); break;
    default: TORCH_CHECK(false, "fill_normal_cuda: dtype must be float32 or float64");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next;
}

// Appendix C's draw: element e of row e / n maps its 64-bit draw through that row's table.
__global__ void choice_kernel(uint64_t *x, uint64_t total, uint64_t n, uint32_t m, const uint64_t *S,
                              const uint64_t *cut, const uint32_t *alias) {
    for (uint64_t e = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; e < total;
         e += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t row = e / n, r = x[e], j = __umul64hi(r, m), c = row * m + j;
        x[e] = __umul64hi(r * m, S[row]) < cut[c] ? j : alias[c];
    }
}

// The plain u64 fill writes the draws into out, and the kernel maps them in place, so out
// holds the indices of the CPU fill with no scratch.
uint64_t fill_choice_cuda(torch::Tensor out, torch::Tensor S, torch::Tensor cut, torch::Tensor alias,
                          const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kLong && out.dim() == 2,
                "fill_choice_cuda: need a contiguous int64 CUDA matrix");
    TORCH_CHECK(S.device() == out.device() && cut.device() == out.device() && alias.device() == out.device(),
                "fill_choice_cuda: the table must be on the device of out");
    c10::cuda::CUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const uint64_t total = (uint64_t)out.numel(), n = (uint64_t)out.size(1);
    uint64_t *x = static_cast<uint64_t *>(out.data_ptr());
    uint64_t next = tandem::fill_u64(key.data(), pos, K, x, total, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (total) {
        const unsigned blocks = (unsigned)std::min<uint64_t>((total + 255) / 256, 1u << 16);
        choice_kernel<<<blocks, 256, 0, stream>>>(x, total, n, (uint32_t)cut.size(1),
                                                  static_cast<const uint64_t *>(S.data_ptr()),
                                                  static_cast<const uint64_t *>(cut.data_ptr()),
                                                  static_cast<const uint32_t *>(alias.data_ptr()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return next;
}

uint64_t fill_exponential_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cuda() && out.is_contiguous(), "fill_exponential_cuda: need a contiguous CUDA tensor");
    c10::cuda::CUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t n = (size_t)out.numel();
    uint64_t next;
    switch (out.scalar_type()) {
    case torch::kFloat: next = tandem::fill_exponential_f32(key.data(), pos, K, out.data_ptr<float>(), n, stream); break;
    case torch::kDouble: next = tandem::fill_exponential_f64(key.data(), pos, K, out.data_ptr<double>(), n, stream); break;
    default: TORCH_CHECK(false, "fill_exponential_cuda: dtype must be float32 or float64");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next;
}
