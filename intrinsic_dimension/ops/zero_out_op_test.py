import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
    def testZeroOut(self):
        zero_out_module = tf.load_op_library('./zero_out.so')
        with self.test_session():
            result = zero_out_module.zero_out([5, 4, 3, 2, 1])
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])


if __name__ == "__main__":
    tf.test.main()
