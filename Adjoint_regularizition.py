import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet


class Regularized(model_correction):
    linear = False
    batch_size = 16
    experiment_name = 'AdjointRegularization'

    # the computational model
    def get_network(self, channels):
        return UNet(channels_out=channels)

    # the loss functionals - can be overwritten in subclasses
    def loss_fct(self, output, true_meas):
        loss = tf.sqrt(tf.reduce_sum(tf.square(output - true_meas), axis=(1, 2, 3)))
        l2 = tf.reduce_mean(loss)
        tf.summary.scalar('Loss_L2', l2)
        return l2

    def __init__(self, path, true_np, appr_np, data_sets):
        super(Regularized, self).__init__(path, data_sets)

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

        # placeholders
        self.approximate_y = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1]],
                                            dtype=tf.float32)
        self.true_y = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1]], dtype=tf.float32)

        # Learning parameters
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # add a channel dimension
        ay = tf.expand_dims(self.approximate_y, axis=3)
        ty = tf.expand_dims(self.true_y, axis=3)

        # the network output
        self.output = self.UNet.net(ay)

        # l2 loss functional
        loss = tf.sqrt(tf.reduce_sum(tf.square(self.output - ty), axis=(1, 2, 3)))
        self.l2 = tf.reduce_mean(loss)
        tf.summary.scalar('Loss_L2', self.l2)

        # adjoint loss

        # compute the direction y_evalute as Psi y-y with y correct measurements
        self.direction = self.UNet.net(ty)-ty
        scalar_prod = tf.reduce_sum(tf.multiply(self.output, self.direction))
        self.gradients = tf.gradients(scalar_prod, self.approximate_y)[0]
        apr_x = multiply(self.gradients, tf.transpose(self.m_appr))
        true_x = multiply(self.direction, tf.transpose(self.m_true))
        self.loss_adj = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(apr_x - true_x), axis=(1, 2))))
        tf.summary.scalar('Loss_Adjoint', self.loss_adj)

        # empiric value to ensure both losses are of the same order
        weighting_factor = 5
        self.total_loss = weighting_factor*self.loss_adj + self.l2
        tf.summary.scalar('TotalLoss', self.total_loss)

        # Optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                             global_step=self.global_step)

        # some tensorboard logging
        tf.summary.image('TrueData', ty, max_outputs=1)
        tf.summary.image('ApprData', ay, max_outputs=1)
        tf.summary.image('NetworkData', self.output, max_outputs=1)
        tf.summary.image('TrueAdjoint', true_x, max_outputs=1)
        tf.summary.image('NetworkAdjoint', apr_x, max_outputs=1)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/', self.sess.graph)

        # initialize variables
        tf.global_variables_initializer().run()

        # load in existing model
        self.load()

    def evaluate(self, y):
        y, change = self.feedable_format(y)
        result = self.sess.run(self.output, feed_dict={self.approximate_y: y})[..., 0]
        if change:
            result = result[0, ...]
        return result

    def differentiate(self, point, direction):
        location, change = self.feedable_format(point)
        direction, _ = self.feedable_format(direction)
        result = self.sess.run(self.gradients, feed_dict={self.approximate_y: location, self.direction: direction})
        if change:
            result = result[0, ...]
        return result

    def train(self, learning_rate):
        appr, true, image = self.data_sets.train.next_batch(self.batch_size)
        self.sess.run(self.optimizer, feed_dict={self.approximate_y: appr, self.true_y: true,
                                                 self.learning_rate: learning_rate})

    def log(self):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        iteration, summary = self.sess.run([self.global_step, self.merged],
                                           feed_dict={self.approximate_y: appr, self.true_y: true})
        self.writer.add_summary(summary, iteration)