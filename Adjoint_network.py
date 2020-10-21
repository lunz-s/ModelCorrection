import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet
import numpy as np
from ut import huber_TV
from matplotlib import pyplot as plt

### Warning ####
# Unlike in the other implementations of the correction opertor,
# here the correction operator already includes the approximate forward operator!

def l2(tensor):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1, 2, 3))))

def l2_batch(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1, 2, 3)))


class TwoNets(model_correction):
    linear = False
    batch_size = 32

    @staticmethod
    def angle(dir1, dir2):
        product = tf.reduce_sum(tf.multiply(dir1, dir2), axis=(1, 2, 3))
        norm_product = tf.multiply(l2_batch(dir1), l2_batch(dir2))
        return tf.reduce_mean(tf.divide(product, norm_product))

    @staticmethod
    def alignement(dirAppprox, dirTrue):
        product = tf.reduce_sum(tf.multiply(dirAppprox, dirTrue), axis=(1, 2, 3))
        norm_product = tf.multiply(l2_batch(dirTrue), l2_batch(dirTrue))
        return tf.reduce_mean(tf.divide(product, norm_product))

    def get_network(self, channels):
        return UNet(channels_out=channels)

    def __init__(self, path, data_sets, true_np, appr_np, experiment_name='TwoNetworks',
                 noise_level=.01, characteristic_scale=.34, lam=0.001, savepoint=None):
        super(TwoNets, self).__init__(path, data_sets, experiment_name=experiment_name)
        # overwrite the input dimension, as the operator already includes the approximation
        self.input_dim = self.image_size
        self.lam = lam
        self.noise_level = noise_level*characteristic_scale

        # Setting up the operators
        self.true_op = true_np
        self.appr_op = appr_np

        # extract matrices for efficient tensorflow implementation during training
        self.m_true = tf.constant(self.true_op.m, dtype=tf.float32)
        self.m_appr = tf.constant(self.appr_op.m, dtype=tf.float32)

        def multiply(tensor, matrix):
            tensor_flipped = tf.reverse(tensor, axis=[1])
            shape = tensor.shape
            reshaped = tf.reshape(tensor_flipped, [-1, shape[1]*shape[2], 1])
            result = tf.tensordot(reshaped, matrix, axes=[[1], [1]])
            return tf.reshape(result, [-1, shape[1], shape[2], 1])

        def multiply_adjoint(tensor, matrix):
            shape = tensor.shape
            reshaped = tf.reshape(tensor, [-1, shape[1]*shape[2], 1])
            prod = tf.tensordot(reshaped, tf.transpose(matrix), axes=[[1], [1]])
            flipped = tf.reverse(tf.reshape(prod, [-1, shape[1], shape[2], 1]), axis=[1])
            return flipped

        ### The forward network ###

        # placeholders
        self.input_image = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)
        self.is_train = tf.placeholder(tf.bool, shape=())

        # the data Term is used to compute the direction for the gradient regularization
        self.data_term = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1], 1], dtype=tf.float32)

        # methode to get the initial guess in tf
        noise = tf.random_normal(shape=tf.shape(self.input_image), mean=0.0, stddev=self.noise_level, dtype=tf.float32)
        self.measurement = multiply(self.input_image, self.m_true) + noise
        self.x_ini = 4.0*multiply_adjoint(self.measurement, self.m_appr)

        # Compute the corresponding measurements with the true and approximate operators
        self.true_y = multiply(self.input_image, self.m_true)
        self.approximate_y = multiply(self.input_image, self.m_appr)

        self.uncorrected_grad = multiply_adjoint(self.approximate_y-self.data_term, self.m_appr)
        self.true_grad = multiply_adjoint(self.true_y-self.data_term, self.m_true)

        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # the network output
        with tf.variable_scope('Forward_Correction'):
            self.output = self.UNet.net(self.approximate_y, is_train=self.is_train)

            # forward loss
            self.l2 = l2(self.output - self.true_y)

            # optimization algorithm
            self.global_step = tf.Variable(0, name='Step', trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2,
                                                                                 global_step=self.global_step)

            # Direction the adjoint correction is calculated in
            self.direction = self.output - self.data_term
            direction = tf.stop_gradient(self.direction)

            # some tensorboard logging
            tf.summary.image('TrueData', self.true_y, max_outputs=1)
            tf.summary.image('ApprData', self.approximate_y, max_outputs=1)
            tf.summary.image('NetworkData', self.output, max_outputs=1)
            tf.summary.scalar('Loss_L2', self.l2)

        self.approximate_x = multiply_adjoint(direction, self.m_appr)
        self.true_x = multiply_adjoint(direction, self.m_true)

        with tf.variable_scope('Adjoint_Correction'):
            self.correct_adj = self.UNet.net(self.approximate_x, is_train=self.is_train)

            # forward loss
            self.l2_adj = l2(self.correct_adj - self.true_x)

            # optimization algorithm
            self.step_adjoint = tf.Variable(0, name='Step', trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer_adjoint = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2_adj,
                                                                                 global_step=self.step_adjoint)
            # some tensorboard logging
            tf.summary.image('TrueAdjoint', self.true_x, max_outputs=1)
            tf.summary.image('ApprAdjoint', self.approximate_x, max_outputs=1)
            tf.summary.image('NetworkAdjoint', self.correct_adj, max_outputs=1)
            tf.summary.scalar('Loss_Adjoint', self.l2_adj)

        # L1 regularization term
        TV = huber_TV(self.input_image)
        self.average_TV = tf.reduce_sum(TV)
        self.TV_grad = tf.gradients(tf.reduce_sum(TV), self.input_image)[0]

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + str(self.lam)+'/Logs/')

        # tracking for while solving the gradient descent over the data term
        with tf.name_scope('DataGradDescent'):
            l = []
            self.ground_truth = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)
            self.rel_loss_ad =  self.l2_adj/l2(self.true_x)
            l.append(tf.summary.scalar('Loss_Adjoint', self.l2_adj))
            l.append(tf.summary.scalar('Relative_Loss_Adjoint', self.rel_loss_ad))
            self.rel_loss_forw = self.l2/l2(self.true_y)
            l.append(tf.summary.scalar('Loss_Forward', self.l2))
            l.append(tf.summary.scalar('Relative_Loss_Forward', self.l2/l2(self.true_y)))
            self.quality = l2(self.input_image - self.ground_truth)
            self.quality_rel = self.quality/l2(self.ground_truth)
            l.append(tf.summary.scalar('Quality', self.quality))
            l.append(tf.summary.scalar('DataTerm_Approx', l2(direction)))

            self.DataTermTrue =  l2(self.true_y-self.data_term)
            l.append(tf.summary.scalar('DataTerm_True', self.DataTermTrue))

            self.DataTermUncorrected = l2(self.approximate_y - self.data_term)
            l.append(tf.summary.scalar('DataTerm_Uncorrected', self.DataTermTrue))

            l.append(tf.summary.scalar('TV_regularization', self.average_TV))

            # Computing the angle
            self.angle_true = self.angle(self.correct_adj, self.true_grad)
            l.append(tf.summary.scalar('Angle', self.angle_true))

            # Uncorrected angle for comparison
            self.angle_uncorrected = self.angle(self.uncorrected_grad, self.true_grad)
            l.append(tf.summary.scalar('Uncorrected_Angle', self.angle_uncorrected))

            # Checking the input variance
            m = l2(self.input_image)
            v = tf.reduce_mean(tf.square(l2_batch(self.input_image) - m))
            l.append(tf.summary.scalar('Mean_Image_Norm', m))
            l.append(tf.summary.scalar('Variance_Image_Norm', v))

            # Computing the average norm of the direction
            l.append(tf.summary.scalar('Direction_Norm', l2(self.direction)))

            l.append(tf.summary.image('True_Data', self.true_y, max_outputs=1))
            l.append(tf.summary.image('Network_Data', self.output, max_outputs=1))
            l.append(tf.summary.image('True_Gradient', self.true_x, max_outputs=1))
            l.append(tf.summary.image('Network_Gradient', self.correct_adj, max_outputs=1))
            l.append(tf.summary.image('GroundTruth', self.ground_truth, max_outputs=1))
            l.append(tf.summary.image('Reconstruction', self.input_image, max_outputs=1))
            self.merged_opt = tf.summary.merge(l)

        # initialize variables
        tf.global_variables_initializer().run()

        # load in existing model
        self.load(savepoint=savepoint)

    def step(self, x, data_term, lam, step_size, positivity=True):
        corrected_grad, true_y, approx_y, corrected_y, approx_grad, true_grad, TV_grad=self.sess.run(
            [self.correct_adj, self.true_y, self.approximate_y, self.output, self.uncorrected_grad, self.true_grad, self.TV_grad],
            feed_dict={self.input_image: x, self.data_term: data_term, self.is_train: False})
        x = self.update(x, corrected_grad, TV_grad, lam=lam, step_size=step_size, positivity=positivity)
        return x, {
            'TrueGrad': true_grad,
            'ApproxGrad': approx_grad,
            'CorrectedGrad': corrected_grad,
            'TrueY': true_y,
            'ApproxY': approx_y,
            'CorrectedY': corrected_y,
        }
        
            
        

    def update(self, image, data_gradient, TV_gradient, lam, step_size, positivity=True):
        grad = data_gradient + lam * TV_gradient
        res = image - 2*step_size*grad
        if positivity:
            return np.maximum(0, res)
        else:
            return res

    def evaluate(self, x):
        pass

    def differentiate(self, point, direction):
        pass

    def train(self, recursion, steps_size, learning_rate, lam=None, augmentation=None, train_every_n=1):
        if lam is None:
            lam = self.lam

        image = self.data_sets.train.next_batch(self.batch_size)
        if not augmentation is None:
            image = augmentation(image)
        x, true = self.sess.run([self.x_ini, self.measurement], feed_dict={self.input_image: image})

        # projecting initial guess
        x = np.maximum(x, 0)

        for k in range(recursion):
            if k % train_every_n == 0:
                self.sess.run(self.optimizer, feed_dict={self.input_image: x, self.data_term: true,
                                                         self.learning_rate: learning_rate, self.is_train: True})

                self.sess.run(self.optimizer_adjoint, feed_dict={self.input_image: x, self.data_term: true,
                                                                 self.learning_rate: learning_rate, self.is_train: True})

            update, tv_grad = self.sess.run([self.correct_adj, self.TV_grad],
                                   feed_dict={self.input_image: x, self.data_term: true,
                                              self.ground_truth: image, self.is_train: False})
            x = self.update(x, update, TV_gradient=tv_grad, lam=lam, step_size=steps_size, positivity=True)

    def log(self, recursions, steps_size):
        image = self.data_sets.test.default_batch(self.batch_size)
        x, true = self.sess.run([self.x_ini, self.measurement], feed_dict={self.input_image: image})

        iteration, summary = self.sess.run([self.global_step, self.merged],
                    feed_dict={self.input_image: x, self.data_term: true, self.is_train: False})
        self.writer.add_summary(summary, iteration)


    def log_optimization(self, image, recursions, step_size, lam, positivity=True, operator='Corrected', verbose=False, tensorboard=True, n_print=5):
        x, true = self.sess.run([self.x_ini, self.measurement], feed_dict={self.input_image: image})

        if positivity:
            x = np.maximum(x, 0)

        if verbose:
            plt.figure(figsize=(18,4))
            plt.subplot(131)
            plt.imshow(true[0,...,0])
            plt.title('Measurement')
            plt.axis('off')
            plt.colorbar()
            plt.subplot(132)
            plt.imshow(image[0,...,0])
            plt.title('True Image')
            plt.axis('off')
            plt.colorbar()
            plt.subplot(133)
            plt.imshow(x[0,...,0])
            plt.title('Backprojection')
            plt.axis('off')
            plt.colorbar()
            plt.show()

        if tensorboard:
            if operator == 'Corrected':
                writer = tf.summary.FileWriter(self.raw_path + 'GradDesc/Lambda_{}/{}'.format(lam, self.experiment_name))
            elif operator == 'True':
                writer = tf.summary.FileWriter(self.raw_path + 'GradDesc/Lambda_{}/GroundTruth'.format(lam))
            elif operator == 'Approx':
                writer = tf.summary.FileWriter(self.raw_path + 'GradDesc/Lambda_{}/ApproxUncorrected'.format(lam))
            else:
                raise ValueError(f'Operator {operator} not supported.')
        #Setting up tracking of main quantities in lists
        res = {
            'quality' : [],
            'angle': [],
            'uncorrectedAngle': [],
            'DataTerm': [],
            'uncorrectedDataTerm': [],
            'loss_fwd': [],
            'rel_loss_fwd': [],
            'loss_adj': [],
            'rel_loss_adj': []

        }
        for k in range(recursions):
            summary, approx_grad, TV_grad, TV_value, quality, dataTerm, dataTermUncor, angle, uncor_angle, \
            true_grad, recon, uncor_grad, loss_fwd, rel_loss_fwd, loss_adj, rel_loss_adj = self.sess.run(
                [self.merged_opt, self.correct_adj, self.TV_grad, self.average_TV, self.quality_rel, self.DataTermTrue,
                 self.DataTermUncorrected, self.angle_true, self.angle_uncorrected, self.true_grad,
                 self.input_image, self.uncorrected_grad, self.l2, self.rel_loss_forw, self.l2_adj, self.rel_loss_ad],
                feed_dict={self.input_image: x, self.data_term: true,
                           self.ground_truth: image, self.is_train: False})

            res['quality'].append(quality)
            res['angle'].append(angle)
            res['uncorrectedAngle'].append(uncor_angle)
            res['DataTerm'].append(dataTerm)
            res['uncorrectedDataTerm'].append(dataTermUncor)
            res['loss_fwd'].append(loss_fwd)
            res['rel_loss_fwd'].append(rel_loss_fwd)
            res['loss_adj'].append(loss_adj)
            res['rel_loss_adj'].append(rel_loss_adj)

            if verbose and k % n_print == 0:
                print(f'Quality {quality}, Data Term {dataTerm}, Regularizer {self.lam*TV_value}, Angle {angle}, Uncor Angle {uncor_angle}')
                # Plotting reconstruction
                plt.figure(figsize=(18,4))
                plt.subplot(151)
                plt.imshow(true_grad[0,...,0])
                plt.colorbar()
                plt.axis('off')
                plt.title('True Gradient')
                plt.subplot(152)
                plt.imshow(approx_grad[0,...,0])
                plt.colorbar()
                plt.axis('off')
                plt.title('Approx Gradient')



                plt.subplot(153)
                plt.imshow(uncor_grad[0,...,0])
                plt.colorbar()
                plt.axis('off')
                plt.title('Uncorrected Gradient')
                plt.subplot(154)
                plt.imshow(self.lam*TV_grad[0,...,0])
                plt.colorbar()
                plt.axis('off')
                plt.title('Huber gradient')
                plt.subplot(155)
                plt.imshow(recon[0,...,0])
                plt.colorbar()
                plt.axis('off')
                plt.title('Reconstruction')
                plt.show()
            if tensorboard:
                writer.add_summary(summary, k)
            # Determening which operator to use for gradient steps
            if operator == 'Corrected':
                gradient = approx_grad
            elif operator == 'True':
                gradient = true_grad
            elif operator == 'Approx':
                gradient = uncor_grad
            else:
                raise ValueError(f'Operator {operator} not supported.')
            x = self.update(x, gradient, TV_grad, lam=lam, step_size=step_size, positivity=positivity)
        if tensorboard:
            writer.flush()
            writer.close()

        return res, x


    def log_optimization_old(self, image, recursions, step_size, lam, positivity = True, training_data=False):
        x, true = self.sess.run([self.x_ini, self.measurement], feed_dict={self.input_image: image})
        if training_data:
            writer = tf.summary.FileWriter(self.raw_path + 'GradDesc/Lambda_{}/TrainData/{}'.format(lam, self.experiment_name))
        else:
            writer = tf.summary.FileWriter(self.raw_path + 'GradDesc/Lambda_{}/{}'.format(lam, self.experiment_name))
        for k in range(recursions):
            summary, data_grad, TV_grad = self.sess.run([self.merged_opt, self.correct_adj, self.TV_grad],
                               feed_dict={self.input_image: x, self.data_term: true,
                                          self.ground_truth: image, self.is_train: False})
            writer.add_summary(summary, k)
            x = self.update(x, data_grad, TV_grad, lam=lam, step_size=step_size, positivity=positivity)
        writer.flush()
        writer.close()

    def log_gt_optimization(self, image, recursions, step_size, lam, positivity=True):
        x, true = self.sess.run([self.x_ini, self.measurement], feed_dict={self.input_image: image})
        writer = tf.summary.FileWriter(self.raw_path + 'GradDesc/Lambda_{}/GroundTruth'.format(lam))
        for k in range(recursions):
            summary, data_grad, TV_grad = self.sess.run([self.merged_opt, self.true_grad, self.TV_grad],
                               feed_dict={self.input_image: x, self.data_term: true,
                                          self.ground_truth: image, self.is_train: False})
            writer.add_summary(summary, k)
            x = self.update(x, data_grad, TV_grad, lam=lam, step_size=step_size, positivity=positivity)
        writer.flush()
        writer.close()

    def log_approx_optimization(self, image, recursions, step_size, lam, positivity=True):
        x, true = self.sess.run([self.x_ini, self.measurement], feed_dict={self.input_image: image})
        writer = tf.summary.FileWriter(self.raw_path + 'GradDesc/Lambda_{}/ApproxUncorrected'.format(lam))
        for k in range(recursions):
            summary, data_grad, TV_grad = self.sess.run([self.merged_opt, self.uncorrected_grad, self.TV_grad],
                               feed_dict={self.input_image: x, self.data_term: true,
                                          self.ground_truth: image, self.is_train: False})
            writer.add_summary(summary, k)
            x = self.update(x, data_grad, TV_grad, lam=lam, step_size=step_size, positivity=positivity)
        writer.flush()
        writer.close()
