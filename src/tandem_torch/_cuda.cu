// CUDA fills over the vendored CUDA header, on the tensor's current stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <cstdint>

#include "tandem.cuh"

using Key = std::array<uint32_t, 4>;

/* 32-bit and 64-bit element types only. Python builds the narrower types and bool from a
 * word fill, since the stream is one byte sequence after alignment. */
uint64_t fill_cuda(torch::Tensor out, const Key &key, uint64_t pos, uint32_t K) {
    TORCH_CHECK(out.device().is_cuda(), "fill_cuda: tensor must be on a CUDA device");
    TORCH_CHECK(out.is_contiguous(), "fill_cuda: tensor must be contiguous");
    c10::cuda::CUDAGuard guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t n = (size_t)out.numel();
    void *p = out.data_ptr();
    uint64_t next;
    switch (out.scalar_type()) {
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
