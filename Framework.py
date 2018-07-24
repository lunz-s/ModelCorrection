import random
import numpy as np
import scipy.ndimage
import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform
import odl
import odl.contrib.tensorflow
from skimage.measure import compare_ssim as ssim

from scipy.misc import imresize
import tensorflow as tf

import Load_PAT2D_data as PATdata
import fastPAT as fpat
from networks import UNet as UNet_class

# abstract class to wrap up the occuring operators in numpy. Can be turned into odl operator using as_odl_operator
class np_operator(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def evaluate(self, y):
        pass

    def differentiate(self, point, direction):
        pass

# methode to compose two numpy operators. Returns operator2 \circ operator1
class concat(np_operator):
    def __init__(self, operator2, operator1):
        self.O1 = operator1
        self.O2 = operator2
        input_dim = operator1.input_dim
        output_dim = operator2.output_dim
        super(concat, self).__init__(input_dim, output_dim)

    def evaluate(self, y):
        return self.O2.evaluate(self.O1.evaluate(y))

    def differentiate(self, point, direction):
        new_point = self.O1.evaluate(point)
        new_direction = self.O2.differentiate(point=new_point, direction=direction)
        return self.O1.differentiate(point=point, direction=new_direction)

### methods to turn numpy operator into corresponding odl operator
class deriv_op_adj(odl.Operator):
    def __init__(self, inp_sp, out_sp, np_model, point):
        self.model = np_model
        self.point = point
        self.out_sp = out_sp
        super(deriv_op_adj, self).__init__(inp_sp, out_sp, linear=True)

    def _call(self, x):
        der = self.model.differentiate(self.point, x)
        return self.out_sp.element(der)

class deriv_op(odl.Operator):
    def __init__(self, inp_sp, out_sp, np_model, point):
        self.model = np_model
        self.point = point
        self.inp_sp = inp_sp
        self.out_sp = out_sp
        super(deriv_op, self).__init__(inp_sp, out_sp, linear=True)

    def _call(self, x):
        return self.out_sp.element(self.model.evaluate(x))

    @property
    def adjoint(self):
        return deriv_op_adj(inp_sp=self.out_sp, out_sp=self.inp_sp, np_model=self.model, point=self.point)

class as_odl_operator(odl.Operator):
    def _call(self, x):
        return self.output_space.element(self.np_model.evaluate(x))

    def derivative(self, point):
        return deriv_op(inp_sp=self.input_space, out_sp=self.output_space, np_model=self.np_model, point=point)

    def __init__(self, np_model):
        self.np_model = np_model
        idim = np_model.input_dim
        left = int(idim[0] / 2)
        right = int(idim[1] / 2)
        self.input_space = odl.uniform_discr([-left, -right], [left, right], [idim[0], idim[1]],
                                             dtype='float32')
        odim = np_model.output_dim
        left = int(odim[0] / 2)
        right = int(odim[1] / 2)
        self.output_space = odl.uniform_discr([-left, -right], [left, right], [odim[0], odim[1]],
                                              dtype='float32')
        super(as_odl_operator, self).__init__(self.input_space, self.output_space)

# PAT as numpy operator
class PAT_operator(np_operator):
    def __init__(self, PAT_OP, input_dim, output_dim):
        self.PAT_OP = PAT_OP
        super(PAT_operator, self).__init__(input_dim, output_dim)

    def evaluate(self, y):
        if len(y.shape) == 3:
            res = np.zeros(shape=(y.shape[0], self.output_dim[0], self.output_dim[1]))
            for k in range(y.shape[0]):
                res[k,...] = self.PAT_OP.kspace_forward(y[k,...])
        elif len(y.shape) == 2:
            res = self.PAT_OP.kspace_forward(y)
        else:
            raise ValueError
        return res

    def differentiate(self, point, direction):
        return self.PAT_OP.kspace_backward(direction)

    def inverse(self, y):
        if len(y.shape) == 3:
            res = np.zeros(shape=(y.shape[0], self.input_dim[0], self.input_dim[1]))
            for k in range(y.shape[0]):
                res[k,...] = self.PAT_OP.kspace_backward(y[k,...])
        elif len(y.shape) == 2:
            res = self.PAT_OP.kspace_backward(y)
        else:
            raise ValueError
        return res

# The model correction as numpy operator
class model_correction(np_operator):
    # makes sure the folders needed for saving the model and logging data are in place
    def generate_folders(self, path):
        paths = {}
        paths['Image Folder'] = path + 'Images'
        paths['Saves Folder'] = path + 'Data'
        paths['Logging Folder'] = path + 'Logs'
        for key, value in paths.items():
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except OSError:
                    pass
                print(key + ' created')

    # save current model to data
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.path+'Data/model', global_step=self.global_step)
        print('Progress saved')

    # load model from data
    def load(self):
        saver = tf.train.Saver()
        if os.listdir(self.path+'Data/'):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        else:
            print('No save found')

    # clears computational graph
    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    # the computational model
    def get_network(self, channels):
        return UNet_class(channels_out=channels)


    def __init__(self, path, measurement_size):
        super(model_correction, self).__init__(measurement_size, measurement_size)
        self.path = path
        self.generate_folders(path)
        self.UNet = self.get_network(channels=1)

        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()

        # placeholders
        self.approximate_y = tf.placeholder(shape=[None, measurement_size[0], measurement_size[1]],
                                            dtype=tf.float32)
        self.true_y = tf.placeholder(shape=[None, measurement_size[0], measurement_size[1]], dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # add a channel dimension
        ay = tf.expand_dims(self.approximate_y, axis=3)
        ty = tf.expand_dims(self.true_y, axis=3)

        # the network output
        self.output = self.UNet.net(ay)

        # loss functional
        self.loss = tf.sqrt(tf.reduce_sum(tf.square(self.output - ty), axis=(1,2,3)))
        self.l2 = tf.reduce_mean(self.loss)

        # optimization algorithm
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2,
                                                                             global_step=self.global_step)

        # some tensorboard logging
        tf.summary.scalar('Loss_L2', self.l2)
        tf.summary.image('TrueData', ty, max_outputs=2)
        tf.summary.image('ApprData', ay, max_outputs=2)
        tf.summary.image('NetworkOutput', self.output, max_outputs=2)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/', self.sess.graph)

        # graph for computing the Hessian of the network in a given direction
        self.direction = tf.placeholder(shape=[None, measurement_size[0], measurement_size[1]],
                                            dtype=tf.float32)
        di = tf.expand_dims(self.direction, axis=3)
        scalar_prod = tf.reduce_sum(tf.multiply(self.output, di))
        self.gradients = tf.gradients(scalar_prod, self.approximate_y)

        # initialize variables
        tf.global_variables_initializer().run()

        # load in existing model
        self.load()

    @staticmethod
    # puts numpy array in form that can be fed into the graph
    def feedable_format(array):
        dim = len(array.shape)
        changed = False
        if dim == 2:
            array  = np.expand_dims(array, axis=0)
            changed = True
        elif not dim == 3:
            raise ValueError
        return array, changed

    def evaluate(self, y):
        y, change = self.feedable_format(y)
        result = self.sess.run(self.output, feed_dict={self.approximate_y: y})[...,0]
        if change:
            result = result[0,...]
        return result

    def differentiate(self, point, direction):
        location, change = self.feedable_format(point)
        direction, _ = self.feedable_format(direction)
        result = self.sess.run(self.gradients, feed_dict={self.approximate_y: location, self.direction: direction})[0]
        if change:
            result = result[0,...]
        return result

    def train(self, true_data, apr_data, learning_rate):
        self.sess.run(self.optimizer, feed_dict={self.approximate_y: apr_data, self.true_y: true_data,
                                                 self.learning_rate:learning_rate})

    def log(self, true_data, apr_data):
        iteration, loss, summary = self.sess.run([self.global_step, self.l2, self.merged],
                      feed_dict={self.approximate_y: apr_data, self.true_y: true_data})
        self.writer.add_summary(summary, iteration)
        print('Iteration: {}, L2Loss: {}'.format(iteration, loss))

# Class that handels the datasets and training
class framework(object):
    # the data set name determines the saving folder and where to look for training data
    data_set_name = 'balls64'
    # categorizes experiments
    experiment_name = 'default_experiment'
    # angular cut off
    angle = 60
    # tv parameter
    tv_param = 0.001

    def __init__(self):
        # finding the correct path extensions for saving models
        name = platform.node()
        path_prefix = ''
        if name == 'motel':
            path_prefix = '/local/scratch/public/sl767/ModelCorrection/'
        else:
            path_prefix = ''

        # setting the training data
        data_path = path_prefix + 'Data/{}/'.format(self.data_set_name)
        train_append = 'trainDataSet.mat'
        test_append = 'testDataSet.mat'
        self.data_sets = PATdata.read_data_sets(data_path + train_append,
                                                data_path + test_append)

        # put folders for the network parameters in place
        self.path = path_prefix + 'Saves/{}/{}/'.format(self.data_set_name, self.experiment_name)

        # get the image and data space sizes
        self.image_size = self.data_sets.train.get_image_resolution()
        self.measurement_size = self.data_sets.train.get_y_resolution()

        # initializing the PAT transform
        kgridBack = fpat.kgrid(data_path + 'kgrid_small.mat')
        kgridForw = fpat.kgrid(data_path + 'kgrid_smallForw.mat')
        self.plain_pat_operator = fpat.fastPAT(kgridBack, kgridForw, self.angle)
        self.pat_operator = PAT_operator(self.plain_pat_operator, self.image_size, self.measurement_size)
        self.odl_pat = as_odl_operator(self.pat_operator)

        # initialize the correction operator
        self.cor_operator = model_correction(self.path, self.measurement_size)
        self.odl_cor = as_odl_operator(self.cor_operator)

    def train_correction(self, steps, batch_size, learning_rate):
        for k in range(steps):
            appr, true, image = self.data_sets.train.next_batch(batch_size)
            new_appr = self.pat_operator.evaluate(image)
            assert new_appr == appr
            self.cor_operator.train(true_data=true, apr_data=appr, learning_rate=learning_rate)
            if k%20 == 0:
                appr, true, image = self.data_sets.test.next_batch(batch_size)
                self.cor_operator.log(true_data=true, apr_data=appr)
        self.cor_operator.save()

    ### methods to run the pdhg algorithm
    @staticmethod
    def _tv_reconstruction(y, start_point, operator, param=0.0001, steps=50):
        space = operator.domain
        ran = operator.range
        # the operators
        gradients = odl.Gradient(space, method='forward')
        broad_op = odl.BroadcastOperator(operator, gradients)
        # define empty functional to fit the chambolle_pock framework
        g = odl.solvers.ZeroFunctional(broad_op.domain)

        # the norms
        l1_norm = param * odl.solvers.L1Norm(gradients.range)
        l2_norm_squared = odl.solvers.L2NormSquared(ran).translated(y)
        functional = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)

        tau = 10.0
        sigma = 0.1
        niter = steps

        # find starting point
        x = space.element(start_point)

        # Run the optimization algoritm
        # odl.solvers.chambolle_pock_solver(x, functional, g, broad_op, tau = tau, sigma = sigma, niter=niter)
        odl.solvers.pdhg(x, functional, g, broad_op, tau=tau, sigma=sigma, niter=niter)
        return x

    def tv_generic(self, data, corrected=True, param=tv_param):
        if corrected:
            operator=self.odl_cor*self.odl_pat
        else:
            operator=self.odl_pat
        if len(data.shape) == 3:
            result = np.zeros(shape=(data.shape[0], self.image_size[0], self.image_size[1]))
            for k in range(data.shape[0]):
                starting_point = self.pat_operator.inverse(data[k,...])
                result[k,...] = self._tv_reconstruction(y=data[k,...], start_point=starting_point,
                                                       operator=operator, param=param)
        else:
            starting_point = self.pat_operator.inverse(data)
            result = self._tv_reconstruction(y=data, start_point=starting_point, operator=operator, param=param)
        return result

    def evaluate_tv(self, param=tv_param, batch_size=10, corrected=True, data=None):
        if data is None:
            appr, true, image = self.data_sets.train.next_batch(batch_size)
        else:
            appr, true, image = data[0], data[1], data[2]
        recon = self.tv_generic(data=true, corrected=corrected, param=param)
        # compute L2 error
        l2 = np.average(np.sqrt(np.sum(np.square(recon-image), axis=(1,2))))
        # comput L2 error with naive methode for comparison
        pseude_inverse = self.pat_operator.inverse(true)
        l2_pi = np.average(np.sqrt(np.sum(np.square(pseude_inverse-image), axis=(1,2))))
        print('Parameter: {}, L2 PseudoInv: {}, L2 Variational: {}'.format(param, l2_pi, l2))
        return l2

    def find_tv_param(self, param_list, corrected=False):
        appr, true, image = self.data_sets.train.next_batch(1)
        for param in param_list:
           self.evaluate_tv(param=param, corrected=corrected, data=[appr, true, image])


