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
    default: TORCH_CHECK(false, "fill_cuda: unsupported dtype ", out.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return next;
}
