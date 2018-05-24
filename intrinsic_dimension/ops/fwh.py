import os
import tensorflow as tf
from tensorflow.python.framework import ops

fwh_so = os.path.join(os.path.dirname(__file__), 'fast_walsh_hadamard.so')

try:
    fast_walsh_hadamard_module = tf.load_op_library(fwh_so)
except tf.errors.NotFoundError:
    print '\n\nError: could not find compiled fast_walsh_hadamard.so file. Tried loading from this location:\n\n    %s\n\nRun "make all" in lab/ops first.\n\n' % fwh_so
    raise

fast_walsh_hadamard = fast_walsh_hadamard_module.fast_walsh_hadamard


@ops.RegisterGradient("FastWalshHadamard")
def _fast_walsh_hadamard_grad(op, grad):
    '''The gradients for `fast_walsh_hadamard`.
  
    Args:
      op: The `fast_walsh_hadamard` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `fast_walsh_hadamard` op.
  
    Returns:
      Gradients with respect to the input of `fast_walsh_hadamard`.
    '''

    gg = fast_walsh_hadamard(grad)
    return [gg]
