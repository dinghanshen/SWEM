import tensorflow as tf
from fwh import fast_walsh_hadamard as c_fast_walsh_hadamard
import numpy as np
import time

class FastWalshHadamardTest(tf.test.TestCase):
    def testFastWalshHadamard(self):
        with self.test_session():
            for i in range(20, 30):
                V = np.random.RandomState(123).randn(2 ** i).astype(np.float32)
                V = tf.constant(V, tf.float32)

                with tf.device('/cpu'):
                    cpu_out = c_fast_walsh_hadamard(V)
                with tf.device('/gpu'):
                    gpu_out = c_fast_walsh_hadamard(V)

                a_start = time.time()
                a = cpu_out.eval();
                a_end = time.time()
                b = gpu_out.eval();
                b_end = time.time()
                print('Size: 2**{} = {} CPU: {} GPU: {} Speedup: {}'.format(i, 2 ** i, a_end-a_start, b_end-a_end, (a_end-a_start)/(b_end-a_end)))
                self.assertAllClose(a, b, rtol=1e-02, atol=1e-02)

if __name__ == "__main__":
    tf.test.main()
