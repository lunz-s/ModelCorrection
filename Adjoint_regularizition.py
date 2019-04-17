import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet
import random

def l2(tensor):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1, 2, 3))))


class Regularized(model_correction):
    linear = False
    batch_size = 16

    ### Weighting factor of 0 corresponds to Forward loss only
    weighting_factor = 1

    # the computational model
    def get_network(self, channels):
        return UNet(channels_out=channels)

    def __init__(self, path, true_np, appr_np, data_sets, experiment_name='AdjointRegularization'):
        super(Regularized, self).__init__(path, data_sets, experiment_name=experiment_name)

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


        # placeholders
        # The location x in image space
        self.input_image = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)

        # the data Term is used to compute the direction for the gradient regularization
        self.data_term = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1], 1], dtype=tf.float32)

        # methode to get the initial guess in tf
        self.x_ini = multiply_adjoint(self.data_term, self.m_true)

        # Compute the corresponding measurements with the true and approximate operators
        self.true_y = multiply(self.input_image, self.m_true)
        self.approximate_y = multiply(self.input_image, self.m_appr)

        self.learning_rate = tf.placeholder(dtype=tf.float32)
        print(self.approximate_y.shape)
        self.output = self.UNet.net(self.approximate_y)

        # The loss caused by forward misfit
        self.l2 = l2(self.output - self.true_y)

        # compute the direction to evaluate the adjoint in
        self.direction = self.output-self.data_term
        direction = tf.stop_gradient(self.direction)

        # direction = self.true_y-self.data_term

        # The loss caused by adjoint misfit
        scalar_prod = tf.reduce_sum(tf.multiply(self.output, direction))
        self.gradients = tf.gradients(scalar_prod, self.approximate_y)[0]
        self.apr_x = multiply_adjoint(self.gradients, self.m_appr)
        self.true_x = multiply_adjoint(direction, self.m_true)
        self.loss_adj = l2(self.apr_x - self.true_x)

        # Overall misfit: Computes the overall l2 misfit between the gradients of the data terms
        self.approx_grad = multiply_adjoint(self.approximate_y-self.data_term, self.m_appr)
        self.true_grad = multiply_adjoint(self.true_y-self.data_term, self.m_true)

        # empiric value to ensure both losses are of the same order
        self.total_loss = self.weighting_factor*self.loss_adj + self.l2
        self.gradient_loss = l2(self.true_grad - self.approx_grad)

        # Optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                             global_step=self.global_step)

        with tf.name_scope('Training'):
            tf.summary.scalar('Loss_Forward', self.l2)
            tf.summary.scalar('Loss_Adjoint', self.loss_adj)
            tf.summary.scalar('TotalLoss', self.total_loss)
            tf.summary.scalar('GradientLoss', self.gradient_loss)
            tf.summary.scalar('Norm_TrueAdjoint', l2(self.true_x))
            tf.summary.scalar('Relative_Loss_Forward', self.l2/l2(self.true_y))
            tf.summary.scalar('Relative_Loss_Adjoint', self.loss_adj/l2(self.true_x))

        # some tensorboard logging
        with tf.name_scope('Data'):
            tf.summary.image('Image', self.input_image, max_outputs=1)
            tf.summary.image('DataTerm', self.data_term, max_outputs=1)
        with tf.name_scope('Forward'):
            tf.summary.image('TrueData', self.true_y, max_outputs=1)
            tf.summary.image('ApprData', self.approximate_y, max_outputs=1)
            tf.summary.image('NetworkData', self.output, max_outputs=1)
        with tf.name_scope('Adjoint'):
            tf.summary.image('TrueAdjoint', self.true_x, max_outputs=1)
            tf.summary.image('NetworkAdjoint', self.apr_x, max_outputs=1)
            tf.summary.image('TrueAdjoint_trueDirection', self.true_grad, max_outputs=1)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/')

        # tracking for while solving the gradient descent over the data term
        with tf.name_scope('DataGradDescent'):
            l = []
            self.ground_truth = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)
            l.append(tf.summary.scalar('Loss_Adjoint', self.loss_adj))
            tf.summary.scalar('Relative_Loss_Adjoint', self.loss_adj/l2(self.true_x))
            l.append(tf.summary.scalar('Loss_Forward', self.l2))
            l.append(tf.summary.scalar('Loss_Gradient', self.gradient_loss))
            self.quality = l2(self.input_image - self.ground_truth)
            l.append(tf.summary.scalar('Quality', self.quality))
            l.append(tf.summary.scalar('DataTerm_Approx', l2(direction)))
            l.append(tf.summary.scalar('DataTerm_True', l2(self.true_y-self.data_term)))
            l.append(tf.summary.image('True_Data', self.true_y, max_outputs=1))
            l.append(tf.summary.image('Network_Data', self.output, max_outputs=1))
            l.append(tf.summary.image('True_Gradient', self.true_x, max_outputs=1))
            l.append(tf.summary.image('Network_Gradient', self.apr_x, max_outputs=1))
            l.append(tf.summary.image('GroundTruth', self.ground_truth, max_outputs=1))
            l.append(tf.summary.image('Reconstruction', self.input_image, max_outputs=1))
            self.merged_opt = tf.summary.merge(l)



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
        # x = image

        for k in range(recursions):
            self.sess.run(self.optimizer, feed_dict={self.input_image: x, self.data_term: true,
                                                     self.learning_rate: learning_rate})
            update = self.sess.run(self.apr_x, feed_dict={self.input_image: x, self.data_term: true,
                                                     self.learning_rate: learning_rate})
            x = x-2*step_size*update

    def log(self, recursions, step_size):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        x = self.sess.run(self.x_ini, feed_dict={self.data_term: true})
        # x = image

        for k in range(1):
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
            summary, update = self.sess.run([self.merged_opt, self.apr_x],
                               feed_dict={self.input_image: x, self.data_term: true,
                                          self.ground_truth: image})
            writer.add_summary(summary, k)
            x = x-2*step_size*update
        writer.flush()
        writer.close()

    def log_gt_optimization(self, recursions, step_size):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        step, x = self.sess.run([self.global_step, self.x_ini], feed_dict={self.data_term: true})
        writer = tf.summary.FileWriter(self.path + 'Logs/GroundTruth')
        for k in range(recursions):
            summary, update = self.sess.run([self.merged_opt, self.true_grad],
                               feed_dict={self.input_image: x, self.data_term: true,
                                          self.ground_truth: image})
            writer.add_summary(summary, k)
            x = x-2*step_size*update
        writer.flush()
        writer.close()

    def log_approx_optimization(self, recursions, step_size):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        step, x = self.sess.run([self.global_step, self.x_ini], feed_dict={self.data_term: true})
        writer = tf.summary.FileWriter(self.path + 'Logs/ApproxUncorrected')
        for k in range(recursions):
            summary, update = self.sess.run([self.merged_opt, self.approx_grad],
                                            feed_dict={self.input_image: x, self.data_term: true,
                                                       self.ground_truth: image})
            writer.add_summary(summary, k)
            x = x - 2 * step_size * update
        writer.flush()
        writer.close()
