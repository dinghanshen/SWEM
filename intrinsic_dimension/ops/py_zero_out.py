import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
zero_out_float = zero_out_module.zero_out_float

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
    """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
    to_zero = op.inputs[0]
    shape = array_ops.shape(to_zero)
    index = array_ops.zeros_like(shape)
    first_grad = array_ops.reshape(grad, [-1])[0]
    to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
    return [to_zero_grad]  # List of one Tensor, since we have one input

@ops.RegisterGradient("ZeroOutFloat")
def _zero_out_float_grad(op, grad):
    """The gradients for `zero_out_float`.

  Args:
    op: The `zero_out_float` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out_float` op.

  Returns:
    Gradients with respect to the input of `zero_out_float`.
  """
    to_zero = op.inputs[0]
    shape = array_ops.shape(to_zero)
    index = array_ops.zeros_like(shape)
    first_grad = array_ops.reshape(grad, [-1])[0]
    to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
    return [to_zero_grad]  # List of one Tensor, since we have one input
