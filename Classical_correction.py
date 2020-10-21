import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet
import numpy as np
from matplotlib import pyplot as plt
from ut import huber_TV


def l2(tensor):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1, 2, 3))))


def l2_batch(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=(1, 2, 3)))


class ClassicalCorrection(model_correction):
    linear = False
    batch_size = 32

    # Abstract not needed as this is not a learned method
    def get_network(self, channels):
        return None

    @staticmethod
    def multiply(tensor, matrix):
        tensor_flipped = tf.reverse(tensor, axis=[1])
        shape = tensor.shape
        reshaped = tf.reshape(tensor_flipped, [-1, shape[1] * shape[2], 1])
        result = tf.tensordot(reshaped, matrix, axes=[[1], [1]])
        return tf.reshape(result, [-1, shape[1], shape[2], 1])

    @staticmethod
    def multiply_adjoint(tensor, matrix):
        shape = tensor.shape
        reshaped = tf.reshape(tensor, [-1, shape[1] * shape[2], 1])
        prod = tf.tensordot(reshaped, tf.transpose(matrix), axes=[[1], [1]])
        flipped = tf.reverse(tf.reshape(prod, [-1, shape[1], shape[2], 1]), axis=[1])
        return flipped

    @staticmethod
    def plain_multiplication(tensor, matrix):
        shape = tensor.shape
        reshaped = tf.reshape(tensor, [-1, shape[1] * shape[2], 1])
        result = tf.tensordot(reshaped, matrix, axes=[[1], [1]])
        return tf.reshape(result, [-1, shape[1], shape[2], 1])

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

    def __init__(self, path, true_np, appr_np, data_sets, lam=.001, noise_level = .01, characteristic_scale = .34,
                 experiment_name='ClassicalRegularization'):
        super(ClassicalCorrection, self).__init__(path, data_sets, experiment_name=experiment_name)

        self.lam = lam
        self.noise_level = noise_level*characteristic_scale

        # Setting up the operators
        self.true_op = true_np
        self.appr_op = appr_np

        # extract matrices for efficient tensorflow implementation during training
        self.m_true = tf.constant(self.true_op.m, dtype=tf.float32)
        self.m_appr = tf.constant(self.appr_op.m, dtype=tf.float32)


        ### Fit the model to data for evaluation.
        mean, self.cov = self.fit_model()
        self.mean = tf.constant(np.expand_dims(np.expand_dims(mean, axis=0), axis=-1), dtype=tf.float32)
        # add in covariance caused by noise
        self.cov_adjusted = self.cov + (self.noise_level**2 * np.identity(64**2))
        self.inv_cov = self.noise_level**2 * (tf.constant(np.linalg.inv(self.cov_adjusted), dtype=tf.float32))

        # placeholders
        # The location x in image space
        self.input_image = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)

        # the data Term is used to compute the direction for the gradient regularization
        self.data_term = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1], 1], dtype=tf.float32)

        # methode to get the initial guess in tf
        noise = tf.random_normal(shape=tf.shape(self.input_image), mean=0.0, stddev=self.noise_level, dtype=tf.float32)
        self.measurement = self.multiply(self.input_image, self.m_true) + noise
        self.x_ini = 4.0*self.multiply_adjoint(self.measurement, self.m_appr)

        # Compute the corresponding measurements with the true and approximate operators
        self.true_y = self.multiply(self.input_image, self.m_true)
        self.true_grad = self.multiply_adjoint(self.true_y-self.data_term, self.m_true)

        # Uncorrected gradient
        self.approx_y = self.multiply(self.input_image, self.m_appr)
        self.uncorrected_grad = self.multiply_adjoint(self.approx_y-self.data_term, self.m_appr)

        # Computing the data term using the corrected approximate model
        self.data_cor = self.approx_y - self.data_term - self.mean
        self.space_weighted = self.plain_multiplication(self.data_cor, self.inv_cov)
        self.approx_grad = self.multiply_adjoint(self.space_weighted, self.m_appr)

        # empiric value to ensure both losses are of the same order
        self.gradient_loss = l2(self.true_grad - self.approx_grad)

        # L1 regularization term
        # TV = tf.image.total_variation(self.input_image)
        TV = huber_TV(self.input_image)
        self.average_TV = tf.reduce_sum(TV)
        self.TV_grad = tf.gradients(tf.reduce_sum(TV), self.input_image)[0]

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + str(self.lam)+'/Logs/')

        # tracking for while solving the gradient descent over the data term
        with tf.name_scope('DataGradDescent'):
            l = []
            self.ground_truth = tf.placeholder(shape=[None, self.image_size[0], self.image_size[1], 1], dtype=tf.float32)
            l.append(tf.summary.scalar('Loss_Gradient', self.gradient_loss))
            self.quality = l2(self.input_image - self.ground_truth)
            self.quality_rel = self.quality/l2(self.ground_truth)
            l.append(tf.summary.scalar('Quality', self.quality))

            self.DataTermTrue = l2(self.true_y - self.data_term)
            l.append(tf.summary.scalar('DataTerm_True', self.DataTermTrue))

            self.DataTermUncorrected = l2(self.approx_y - self.data_term)
            l.append(tf.summary.scalar('DataTerm_Uncorrected', self.DataTermTrue))

            # Computing the angle
            self.angle_true = self.angle(self.approx_grad, self.true_grad)
            l.append(tf.summary.scalar('Angle', self.angle_true))

            # Uncorrected angle for comparison
            self.angle_uncorrected = self.angle(self.uncorrected_grad, self.true_grad)
            l.append(tf.summary.scalar('Uncorrected_Angle', self.angle_uncorrected))

            # Checking the input variance
            m = l2(self.input_image)
            v = tf.reduce_mean(tf.square(l2_batch(self.input_image) - m))
            l.append(tf.summary.scalar('Mean_Image_Norm', m))
            l.append(tf.summary.scalar('Variance_Image_Norm', v))

            # Adding some images for visualization
            l.append(tf.summary.image('True_Gradient', self.true_grad, max_outputs=1))
            l.append(tf.summary.image('Network_Gradient', self.approx_grad, max_outputs=1))
            l.append(tf.summary.image('GroundTruth', self.ground_truth, max_outputs=1))
            l.append(tf.summary.image('Reconstruction', self.input_image, max_outputs=1))
            self.merged_opt = tf.summary.merge(l)

        # initialize variables
        tf.global_variables_initializer().run()


    def update(self, image, data_gradient, TV_gradient, lam, step_size, positivity=True):
        grad = data_gradient + lam * TV_gradient
        res = image - 2*step_size*grad
        if positivity:
            return np.maximum(0, res)
        else:
            return res

    # Methods not needed for classical correction
    def evaluate(self, y):
        pass

    def differentiate(self, point, direction):
        pass

    def evaluate_operator(self, x, name='True', adjoint=False):
        inp = tf.placeholder(shape=x.shape, dtype=tf.float32)
        if name == 'True':
            m = self.m_true
        elif name == 'Approx':
            m = self.m_appr
        else:
            raise ValueError(f'Name {name} not supported. Only True and Approx supported')
        if adjoint:
            res = self.multiply_adjoint(inp, m)
        else:
            res = self.multiply(inp, m)
        return self.sess.run(res, feed_dict={inp:x})

    def fit_model(self):
        train_data = np.expand_dims(self.data_sets.train.data, axis=-1)
        batch = train_data.shape[0]
        approx_data = self.evaluate_operator(train_data, name='Approx')
        true_data = self.evaluate_operator(train_data, name='True')
        error = (true_data - approx_data)
        error_flat = np.reshape(error, (batch, 64 * 64))
        mean_error = error_flat.mean(axis=0)
        cov_raw = np.matmul(np.transpose(error_flat), error_flat) / (batch - 1)
        mean_error_exp = np.expand_dims(mean_error, axis=1)
        cov_renormed = cov_raw - np.matmul(mean_error_exp, np.transpose(mean_error_exp))
        return mean_error.reshape(64,64), cov_renormed

    def log_optimization(self, image, recursions, step_size, lam, positivity=True, operator='Corrected', verbose=False, tensorboard=True, n_print=5):
        x, true = self.sess.run([self.x_ini, self.measurement], feed_dict={self.input_image: image})

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

        if positivity:
            x = np.maximum(x, 0)
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
            'uncorrectedDataTerm': []
        }
        for k in range(recursions):
            summary, TV_grad, TV_value, quality, dataTerm, dataTermUncor, angle, uncor_angle, \
            true_grad, approx_grad, recon, uncor_grad = self.sess.run(
                [self.merged_opt, self.TV_grad, self.average_TV, self.quality_rel, self.DataTermTrue,
                 self.DataTermUncorrected, self.angle_true, self.angle_uncorrected, self.true_grad,
                 self.approx_grad, self.input_image, self.uncorrected_grad],
                feed_dict={self.input_image: x, self.data_term: true,
                           self.ground_truth: image})

            res['quality'].append(quality)
            res['angle'].append(angle)
            res['uncorrectedAngle'].append(uncor_angle)
            res['DataTerm'].append(dataTerm)
            res['uncorrectedDataTerm'].append(dataTermUncor)

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