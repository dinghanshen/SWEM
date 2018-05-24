#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "fast_walsh_hadamard.h"

using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("FastWalshHadamard")
.Input("hadam_input: float")
.Output("hadam_output: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

template <typename T>
struct FastWalshHadamardFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const int NN, const int halfLL, const T* input, T* output_flat) {
    // First copy input to output
    for (int ii = 0; ii < NN; ii++) {
      output_flat[ii] = input[ii];
    }

    // Apply Unnormalized Fast Walsh Hadamard transform
    int stride = halfLL;
    int segments = 0;
    float tmp = 0;
    float normalizer = 1.0;
    float sqrt2inv = 0.70710678118654746;
    while (stride >= 1) {
      int cc = 0;
      segments = halfLL / stride;
      for (int seg = 0; seg < segments; seg++) {
        for (int ii = 0; ii < stride; ii++) {
          tmp = output_flat[cc];
          output_flat[cc] = output_flat[cc] + output_flat[cc + stride];
          output_flat[cc + stride] = tmp - output_flat[cc + stride];
          cc += 1;
        }
        cc += stride;
      }
      stride /= 2;
      normalizer *= sqrt2inv;
    }
    
    // Apply normalizing divisor
    for (int ii = 0; ii < NN; ii++) {
      output_flat[ii] = output_flat[ii] * normalizer;
    }
  }
};

template <typename Device, typename T>
class FastWalshHadamardOp : public OpKernel {
public:
  explicit FastWalshHadamardOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));


    // Assumptions: NN is a power of two. Figure out ll such that 2^ll = LL
    const int NN = input_tensor.NumElements();
    int ll = 0;
    int LL = 1;
    while (LL < NN) {
      ll += 1;
      LL *= 2;
    }
    const int halfLL = LL / 2;

    OP_REQUIRES(context, LL == NN,
                errors::InvalidArgument("Expected input with length a power of two. But len is ", NN, ". Maybe pad to ", NN, "."));
    OP_REQUIRES(context, halfLL * 2 == LL,
                errors::InvalidArgument("logic error"));
    FastWalshHadamardFunctor<Device, float>()(
        context->eigen_device<Device>(),
        NN,
        halfLL,
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

REGISTER_KERNEL_BUILDER(Name("FastWalshHadamard").Device(DEVICE_CPU), FastWalshHadamardOp<CPUDevice, float>);

#ifdef GOOGLE_CUDA
// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct FastWalshHadamardFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, const int NN, const int halfLL, const T* in, T* out) {
    FastWalshHadamardKernelLauncher(NN, halfLL, in, out);
  }
};
REGISTER_KERNEL_BUILDER(Name("FastWalshHadamard").Device(DEVICE_GPU), FastWalshHadamardOp<GPUDevice, float>);
#endif
