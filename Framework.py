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

class PAT_operator(np_operator):
    def __init__(self, PAT_OP, input_dim, output_dim):
        self.PAT_OP = PAT_OP
        super(PAT_operator, self).__init__(input_dim, output_dim)

    def evaluate(self, y):
        return self.PAT_OP.kspace_forward(y)

    def differentiate(self, point, direction):
        return self.PAT_OP.kspace_backward(direction)


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


    def evaluate(self, y):
        return self.sess.run(self.output, feed_dict={self.approximate_y: y})

    def differentiate(self, location, direction):
        return self.sess.run(self.gradients, feed_dict={self.approximate_y: location, self.direction: direction})

    def train(self, true_data, apr_data, learning_rate):
        self.sess.run(self.optimizer, feed_dict={self.approximate_y: apr_data, self.true_y: true_data,
                                                 self.learning_rate:learning_rate})

    def log(self, true_data, apr_data):
        iteration, loss, summary = self.sess.run([self.global_step, self.l2, self.merged],
                      feed_dict={self.approximate_y: apr_data, self.true_y: true_data})
        self.writer.add_summary(summary, iteration)
        print('Iteration: {}, L2Loss: {}'.format(iteration, loss))

# set up the odl operator classes we need
class deriv_op_adj(odl.Operator):
    def __init__(self, inp_sp, out_sp, np_model, point):
        self.model = np_model
        self.point = point
        self.out_sp = out_sp
        super(deriv_op_adj, self).__init__(inp_sp, out_sp)

    def _call(self, x):
        der = self.model.differentiate(self.point, x)
        return self.out_sp.element(der)

class deriv_op(odl.Operator):
    def __init__(self, inp_sp, out_sp, np_model, point):
        self.model = np_model
        self.point = point
        self.inp_sp = inp_sp
        self.out_sp = out_sp
        super(deriv_op, self).__init__(inp_sp, out_sp)

    def _call(self, x):
        return self.out_sp.element(self.model.evaluate(x))

    @property
    def adjoint(self):
        return deriv_op_adj(inp_sp=self.out_sp, out_sp=self.inp_sp, np_model=self.model, point=self.point)


# get the learned model correction as odl operator
class as_odl_operator(odl.Operator):
    def _call(self, x):
        return self.output_space.element(self.np_model.evaluate(x))

    def derivative(self, point):
        return deriv_op(inp_sp=self.input_space, out_sp=self.output_space, np_model=self.np_model, point=point)

    def __init__(self, np_model):
        self.np_model = np_model
        idim = np_model.input_dim
        left = int(idim[0]/2)
        right = int(idim[1]/2)
        self.input_space = odl.uniform_discr([-left, -right], [left, right], [idim[0], idim[1]],
                                  dtype='float32')
        odim = np_model.output_dim
        left = int(odim[0]/2)
        right = int(odim[1]/2)
        self.output_space = odl.uniform_discr([-left, -right], [left, right], [odim[0], odim[1]],
                                  dtype='float32')
        super(as_odl_operator, self).__init__(self.input_space, self.output_space)


# This class handling the model approximation and training it
class framework(object):
    # the data set name determines the saving folder and where to look for training data
    data_set_name = 'balls64'
    # categorizes experiments
    experiment_name = 'default_experiment'
    # angular cut off
    angle = 60

    def __init__(self):
        # finding the correct path extensions for saving models
        name = platform.node()
        path_prefix = ''
        if name == 'motel':
            path_prefix =  '/local/scratch/public/sl767/ModelCorrection/'
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
        operator = fpat.fastPAT(kgridBack, kgridForw, self.angle)
        self.pat_operator = PAT_operator(operator, self.image_size, self.measurement_size)


        # initialize the correction operator
        self.cor_operator = model_correction(self.path, self.measurement_size)

    def train_correction(self, steps, batch_size, learning_rate):
        for k in range(steps):
            appr, true, image = self.data_sets.train.next_batch(batch_size)
            self.cor_operator.train(true_data=true, apr_data=appr, learning_rate=learning_rate)
            if k%20 == 0:
                appr, true, image = self.data_sets.test.next_batch(batch_size)
                self.cor_operator.log(true_data=true, apr_data=appr)
        self.cor_operator.save()