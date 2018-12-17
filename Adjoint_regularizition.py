import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet
import random


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
        # The location x in image space
        self.input_image = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)

        # the data Term is used to compute the direction for the gradient regularization
        self.data_term = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1], 1], dtype=tf.float32)

        # methode to get the initial guess in tf
        self.x_ini = multiply(self.data_term, tf.transpose(self.m_true))

        # Compute the corresponding measurements with the true and approximate operators
        self.true_y = multiply(self.input_image, self.m_true)
        self.approximate_y = multiply(self.input_image, self.m_appr)

        # Learning parameters
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # the network output
        self.output = self.UNet.net(self.approximate_y)

        # l2 loss functional
        loss = tf.sqrt(tf.reduce_sum(tf.square(self.output - self.true_y), axis=(1, 2, 3)))
        self.l2 = tf.reduce_mean(loss)
        tf.summary.scalar('Loss_L2', self.l2)

        # compute the direction to evaluate the adjoint in
        self.direction = self.output-self.data_term

        # adjoint computation
        scalar_prod = tf.reduce_sum(tf.multiply(self.output, self.direction))
        self.gradients = tf.gradients(scalar_prod, self.approximate_y)[0]
        self.apr_x = multiply(self.gradients, tf.transpose(self.m_appr))
        self.true_x = multiply(self.direction, tf.transpose(self.m_true))
        self.loss_adj = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.apr_x - self.true_x), axis=(1, 2))))
        tf.summary.scalar('Loss_Adjoint', self.loss_adj)

        # empiric value to ensure both losses are of the same order
        weighting_factor = 7
        self.total_loss = weighting_factor*self.loss_adj + self.l2
        tf.summary.scalar('TotalLoss', self.total_loss)

        # Optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                             global_step=self.global_step)

        # some tensorboard logging
        with tf.name_scope('Forward'):
            tf.summary.image('TrueData', self.true_y, max_outputs=1)
            tf.summary.image('ApprData', self.approximate_y, max_outputs=1)
            tf.summary.image('NetworkData', self.output, max_outputs=1)
        with tf.name_scope('Adjoint'):
            tf.summary.image('TrueAdjoint', self.true_x, max_outputs=1)
            tf.summary.image('NetworkAdjoint', self.apr_x, max_outputs=1)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/')

        # tracking for while solving the gradient descent over the data term
        with tf.name_scope('DataGradDescent'):
            self.ground_truth = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)
            adj = tf.summary.scalar('LossAdjoint', self.loss_adj)
            forward = tf.summary.scalar('LossForward', self.l2)
            quality = tf.summary.scalar('Quality', tf.nn.l2_loss(self.input_image - self.ground_truth))
            self.merged_optimization = tf.summary.merge([adj, forward, quality])


        # initialize variables
        tf.global_variables_initializer().run()

        # load in existing model
        self.load()

    def evaluate(self, y):
        y, change = self.feedable_format(y)
        result = self.sess.run(self.output, feed_dict={self.approximate_y: y})
        if change:
            result = result[0, ..., 0]
        return result

    def differentiate(self, point, direction):
        location, change = self.feedable_format(point)
        direction, _ = self.feedable_format(direction)
        result = self.sess.run(self.gradients, feed_dict={self.approximate_y: location, self.direction: direction})
        if change:
            result = result[0, ...]
        return result

    def train(self, recursions, step_size, learning_rate):
        appr, true, image = self.data_sets.train.next_batch(self.batch_size)
        x = self.sess.run(self.x_ini, feed_dict={self.data_term: true})
        for k in range(recursions):
            _, update = self.sess.run([self.optimizer, self.apr_x], feed_dict={self.input_image: x, self.data_term: true,
                                                     self.learning_rate: learning_rate})
            x = x-2*step_size*update


    def log(self, recursions, step_size):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        x = self.sess.run(self.x_ini, feed_dict={self.data_term: true})
        for k in range(random.randint(0,recursions+1)):
            update = self.sess.run(self.apr_x, feed_dict={self.input_image: x, self.data_term: true})
            x = x-2*step_size*update
        iteration, summary = self.sess.run([self.global_step, self.merged],
                                           feed_dict={self.input_image: x, self.data_term: true})
        self.writer.add_summary(summary, iteration)

    def log_optimization(self, recursions, step_size):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        step, x = self.sess.run([self.global_step, self.x_ini], feed_dict={self.data_term: true})
        writer = tf.summary.FileWriter(self.path + 'Logs/Iteration_' + str(step)+'/')
        for k in range(recursions):
            summary, update = self.sess.run([self.merged_optimization, self.apr_x],
                                   feed_dict={self.input_image: x, self.data_term: true ,
                                              self.ground_truth: image})
            writer.add_summary(summary, k)
            x = x-2*step_size*update
