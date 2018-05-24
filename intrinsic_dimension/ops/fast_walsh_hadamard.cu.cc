#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include <iostream>
#include "fast_walsh_hadamard.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;
using namespace std;

#define EIGEN_USE_GPU


template <typename T>
__global__ void FastWalshHadamardKernel(const int stride, const T* in, T* out) {
    const auto idx = (threadIdx.x + blockIdx.x * blockDim.x);
    const auto elemIdx = (idx / stride ) * (2 * stride) + (idx % stride);
    const auto tmp = in[elemIdx], tmp2 = in[elemIdx + stride];
    out[elemIdx] = tmp + tmp2;
    out[elemIdx + stride] = tmp - tmp2;
}

template <typename T>
__global__ void FastWalshHadamardSubKernel(const T scalar, T* out) {
    const auto idx = (threadIdx.x + blockIdx.x * blockDim.x);
    out[idx] *= scalar;
}


template <typename T>
void FastWalshHadamardKernelLauncher(const int NN, const int halfLL, const T* in, T* out) {
    // Apply Unnormalized Fast Walsh Hadamard transform
    int stride = halfLL;
    float normalizer = 1.0;
    float sqrt2inv = 0.70710678118654746;
    while (stride >= 1) {
      if(stride == halfLL)
          FastWalshHadamardKernel<T><<<max(1, halfLL/256), min(256, halfLL)>>>(stride, in, out);
      else
          FastWalshHadamardKernel<T><<<max(1, halfLL/256), min(256, halfLL)>>>(stride, out, out);

      stride /= 2;
      normalizer *= sqrt2inv;
    }
    FastWalshHadamardSubKernel<T><<<max(1, NN/256), min(256, NN)>>>(normalizer, out);
}

template void FastWalshHadamardKernelLauncher<float>(const int NN, const int halfLL, const float* in, float* out);

#endif
