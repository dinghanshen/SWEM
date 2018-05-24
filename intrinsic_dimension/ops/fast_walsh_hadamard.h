// fast_walsh_hadamard.h
#ifndef KERNEL_FAST_WALSH_HADAMARD_H_
#define KERNEL_FAST_WALSH_HADAMARD_H_

template <typename Device, typename T>
struct FastWalshHadamardFunctor {
  void operator()(const Device& d, const int NN, const int halfLL, const T* input, T* output_flat);
};

#ifdef GOOGLE_CUDA
    template <typename T>
    void FastWalshHadamardKernelLauncher(const int NN, const int halfLL, const T* in, T* out);
#endif


#endif
