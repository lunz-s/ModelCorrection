import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet

### Warning ####
# Unlike in the other implementations of the correction opertor,
# here the correction operator already includes the approximate forward operator!

def l2(tensor):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1, 2, 3))))


class TwoNets(model_correction):
    linear = False
    batch_size = 16
    experiment_name = 'TwoNetworks'

    def get_network(self, channels):
        return UNet(channels_out=channels)

    def __init__(self, path, data_sets, true_np, appr_np):
        super(TwoNets, self).__init__(path, data_sets)
        # overwrite the input dimension, as the operator already includes the approximation
        self.input_dim = self.image_size

        # Setting up the operators
        self.true_op = true_np
        self.appr_op = appr_np

        # extract matrices for efficient tensorflow implementation during training
        self.m_true = tf.constant(self.true_op.m, dtype=tf.float32)
        self.m_appr = tf.constant(self.appr_op.m, dtype=tf.float32)

        def multiply(tensor, matrix):
            tensor_flipped = tf.reverse(tensor, axis=[1])
            shape = tf.shape(tensor)
            reshaped = tf.reshape(tensor_flipped, [-1, shape[1]*shape[2], 1])
            result = tf.tensordot(reshaped, matrix, axes=[[1], [1]])
            return tf.reshape(result, [-1, shape[1], shape[2], 1])

        def multiply_adjoint(tensor, matrix):
            shape = tf.shape(tensor)
            reshaped = tf.reshape(tensor, [-1, shape[1]*shape[2], 1])
            prod = tf.tensordot(reshaped, tf.transpose(matrix), axes=[[1], [1]])
            flipped = tf.reverse(tf.reshape(prod, [-1, shape[1], shape[2], 1]), axis=[1])
            return flipped

        ### The forward network ###

        # placeholders
        self.input_image = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)

        # the data Term is used to compute the direction for the gradient regularization
        self.data_term = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1], 1], dtype=tf.float32)

        # methode to get the initial guess in tf
        self.x_ini = multiply_adjoint(self.data_term, self.m_true)

        # Compute the corresponding measurements with the true and approximate operators
        self.true_y = multiply(self.input_image, self.m_true)
        self.approximate_y = multiply(self.input_image, self.m_appr)

        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # the network output
        with tf.variable_scope('Forward_Correction'):
            self.output = self.UNet.net(self.approximate_y)

            # forward loss
            self.l2 = l2(self.output - self.true_y)

            # optimization algorithm
            self.global_step = tf.Variable(0, name='Step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2,
                                                                                 global_step=self.global_step)

            # Direction the adjoint correction is calculated in
            self.direction = self.output - self.true_y
            direction = tf.stop_gradient(self.direction)

            # some tensorboard logging
            tf.summary.image('TrueData', self.true_y, max_outputs=1)
            tf.summary.image('ApprData', self.approximate_y, max_outputs=1)
            tf.summary.image('NetworkData', self.output, max_outputs=1)
            tf.summary.scalar('Loss_L2', self.l2)

        self.approximate_x = multiply(direction, tf.transpose(self.m_appr))
        self.true_x = multiply(direction, tf.transpose(self.m_true))

        with tf.variable_scope('Adjoint_Correction'):
            self.correct_adj = self.UNet.net(self.approximate_x)

            # forward loss
            self.l2_adj = l2(self.correct_adj - self.true_x)

            # optimization algorithm
            self.step_adjoint = tf.Variable(0, name='Step', trainable=False)
            self.optimizer_adjoint = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2_adj,
                                                                                 global_step=self.step_adjoint)
            # some tensorboard logging
            tf.summary.image('TrueAdjoint', self.true_x, max_outputs=1)
            tf.summary.image('ApprAdjoint', self.approximate_x, max_outputs=1)
            tf.summary.image('NetworkAdjoint', self.correct_adj, max_outputs=1)
            tf.summary.scalar('Loss_Adjoint', self.l2_adj)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/', self.sess.graph)

        self.ground_truth = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)
        self.quality = l2(self.input_image - self.ground_truth)
        tf.summary.scalar('Quality', self.quality)
        tf.summary.scalar('DataTermTrueOP', self.l2(self.true_y - self.data_term))
        tf.summary.scalar('DataTermCorrectedOP', self.l2(self.output - self.data_term))
        self.merged_gd = tf.summary.merge_all()

        # initialize variables
        tf.global_variables_initializer().run()

        # load in existing model
        self.load()

    def evaluate(self, x):
        pass

    def differentiate(self, point, direction):
        pass

    def train(self, recursion, steps_size, learning_rate):
        appr, true, image = self.data_sets.train.next_batch(self.batch_size)
        x = self.sess.run(self.x_ini, feed_dict={self.data_term: true})

        self.sess.run(self.optimizer, feed_dict={self.input_image: x, self.data_term: true,
                                                 self.learning_rate: learning_rate})

        self.sess.run(self.optimizer_adjoint, feed_dict={self.input_image: x, self.data_term: true,
                                                         self.learning_rate: learning_rate})

    def log(self, recursions, steps_size):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        x = self.sess.run(self.x_ini, feed_dict={self.data_term: true})

        iteration, summary = self.sess.run([self.global_step, self.merged],
                    feed_dict={self.input_image: x, self.data_term: true})
        self.writer.add_summary(summary, iteration)

    def log_optimization(self, recursions, step_size):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        step, x = self.sess.run([self.global_step, self.x_ini], feed_dict={self.data_term: true})
        writer = tf.summary.FileWriter(self.path + 'Logs/Iteration_' + str(step)+'/')
        for k in range(recursions):
            summary, update = self.sess.run([self.merged_gd, self.correct_adj],
                               feed_dict={self.input_image: x, self.data_term: true,
                                          self.ground_truth: image})
            writer.add_summary(summary, k)
            x = x-2*step_size*update
        writer.flush()
        writer.close()
