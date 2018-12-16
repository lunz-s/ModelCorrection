import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet

### Warning ####
# Unlike in the other implementations of the correction opertor,
# here the correction operator already includes the approximate forward operator!


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
            shape = tf.shape(tensor)
            reshaped = tf.reshape(tensor, [-1, shape[1]*shape[2], 1])
            result = tf.tensordot(reshaped, matrix, axes=[[1], [0]])
            return tf.reshape(result, [-1, shape[1], shape[2], 1])

        ### The forward network ###

        # placeholders
        self.approximate_y = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1],1],
                                            dtype=tf.float32)
        self.true_y = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1],1], dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # the network output
        with tf.variable_scope('Forward_Correction'):
            self.output = self.UNet.net(self.approximate_y)

            # forward loss
            loss = tf.sqrt(tf.reduce_sum(tf.square(self.output - self.true_y), axis=(1,2,3)))
            self.l2 = tf.reduce_mean(loss)

            # optimization algorithm
            self.global_step = tf.Variable(0, name='Step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2,
                                                                                 global_step=self.global_step)

            # Direction the adjoint correction is calculated in
            self.direction = self.UNet.net(self.true_y) - self.true_y

        # some tensorboard logging
        tf.summary.image('TrueData', self.true_y, max_outputs=1)
        tf.summary.image('ApprData', self.approximate_y, max_outputs=1)
        tf.summary.image('NetworkData', self.output, max_outputs=1)
        tf.summary.scalar('Loss_L2', self.l2)

        # placeholders
        self.approximate_x = multiply(self.direction, tf.transpose(self.m_appr))
        self.true_x = multiply(self.direction, tf.transpose(self.m_true))

        with tf.variable_scope('Adjoint_Correction'):
            self.correct_adj = self.UNet.net(self.approximate_x)

            # forward loss
            loss = tf.sqrt(tf.reduce_sum(tf.square(self.correct_adj - self.true_x), axis=(1,2,3)))
            self.l2_adj = tf.reduce_mean(loss)

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

        # initialize variables
        tf.global_variables_initializer().run()

        # load in existing model
        self.load()

    def evaluate(self, x):
        y, change = self.feedable_format(x)
        result = self.sess.run(self.output, feed_dict={self.approximate_y: y})
        if change:
            result = result[0,...]
        return result

    def differentiate(self, point, direction):
        location, change = self.feedable_format(point)
        direction, _ = self.feedable_format(direction)
        result = self.sess.run(self.correct_adj, feed_dict={self.approximate_x: location,
                                                            self.direction: direction})
        if change:
            result = result[0,...,0]
        return result

    def train_forward(self, learning_rate):
        appr, true, image = self.data_sets.train.next_batch(self.batch_size)
        self.sess.run(self.optimizer, feed_dict={self.approximate_y: appr, self.true_y: true,
                                                 self.learning_rate: learning_rate})

    def train_adjoint(self, learning_rate):
        appr, true, image = self.data_sets.train.next_batch(self.batch_size)
        self.sess.run(self.optimizer_adjoint, feed_dict={self.approximate_y: appr, self.true_y: true,
                                                 self.learning_rate: learning_rate})

    def train(self, learning_rate):
        appr, true, image = self.data_sets.train.next_batch(self.batch_size)
        self.sess.run(self.optimizer, feed_dict={self.approximate_y: appr, self.true_y: true,
                                                 self.learning_rate: learning_rate})
        self.sess.run(self.optimizer_adjoint, feed_dict={self.approximate_y: appr, self.true_y: true,
                                                 self.learning_rate: learning_rate})

    def log(self):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        iteration, summary = self.sess.run([self.global_step, self.merged],
                      feed_dict={self.approximate_y: appr, self.true_y: true})
        self.writer.add_summary(summary, iteration)
