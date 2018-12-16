import numpy as np
import os
import platform
import h5py
import odl
from abc import ABC, abstractmethod
import tensorflow as tf
from Operators import Load_PAT2D_data as PATdata, fastPAT_withAdjoint as fpat

# abstract class to wrap up the occuring operators in numpy. Can be turned into odl operator using as_odl_operator
class np_operator(ABC):
    linear = True

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def evaluate(self, y):
        pass

    @abstractmethod
    def differentiate(self, point, direction):
        pass

#### methods to turn numpy operator into corresponding odl operator
class deriv_op_adj(odl.Operator):
    def __init__(self, inp_sp, out_sp, np_model, point):
        self.linear=np_model.linear
        self.model = np_model
        self.point = point
        self.out_sp = out_sp
        super(deriv_op_adj, self).__init__(inp_sp, out_sp, linear=True)

    def _call(self, x):
        der = self.model.differentiate(self.point, x)
        return self.out_sp.element(der)

    @property
    def adjoint(self):
        if not self.linear:
            raise TypeError('Non linear operators do not admit adjoint')
        else:
            return as_odl_operator(self.model)

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
        self.linear = np_model.linear
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

    @property
    def adjoint(self):
        if not self.linear:
            raise TypeError('Non linear operators do not admit adjoint')
        else:
            return deriv_op_adj(inp_sp=self.input_space, out_sp=self.output_space, np_model=self.np_model, point=0)

# the approximated PAT operator as np_operator
class approx_PAT_operator(np_operator):
    def __init__(self, PAT_OP, input_dim, output_dim):
        self.PAT_OP = PAT_OP
        super(approx_PAT_operator, self).__init__(input_dim, output_dim)

    def evaluate(self, y):
        if len(y.shape) == 4:
            res = np.zeros(shape=(y.shape[0], self.output_dim[0], self.output_dim[1], 1))
            for k in range(y.shape[0]):
                res[k,...,0] = self.PAT_OP.kspace_forward(y[k,...,0])
        elif len(y.shape) == 2:
            res = self.PAT_OP.kspace_forward(y)
        else:
            raise ValueError
        return res

    def differentiate(self, point, direction):
        return self.PAT_OP.kspace_adjoint(direction)

    def inverse(self, y):
        if len(y.shape) == 4:
            res = np.zeros(shape=(y.shape[0], self.input_dim[0], self.input_dim[1], 1))
            for k in range(y.shape[0]):
                res[k,...,0] = self.PAT_OP.kspace_inverse(y[k,...,0])
        elif len(y.shape) == 2:
            res = self.PAT_OP.kspace_inverse(y)
        else:
            raise ValueError
        return res

class approx_PAT_matrix(np_operator):
    def __init__(self, matrix_path, input_dim, output_dim):
        fData = h5py.File(matrix_path, 'r')
        inData = fData.get('Aapprox')
        rows = inData.shape[0]
        cols = inData.shape[1]
        print(rows, cols)
        self.m = np.matrix(inData)
        self.input_sq = input_dim[0]*input_dim[1]
        self.output_sq = output_dim[0]*output_dim[1]
        super(approx_PAT_matrix, self).__init__(input_dim, output_dim)

    # computes the matrix multiplication with matrix
    def evaluate(self, y):
        y = np.flipud(np.asarray(y))
        if len(y.shape) == 4:
            res = np.zeros(shape=(y.shape[0], self.output_dim[0], self.output_dim[1], 1))
            for k in range(y.shape[0]):
                res[k,..., 0] = np.reshape(np.matmul(self.m, np.reshape(y[k,..., 0], self.input_sq)),
                                        [self.output_dim[0], self.output_dim[1]])
        elif len(y.shape) == 2:
            res = np.reshape(np.matmul(self.m, np.reshape(y, self.input_sq)), [self.output_dim[0], self.output_dim[1]])
        else:
            raise ValueError
        return res

    # matrix multiplication with the adjoint of the matrix
    def differentiate(self, point, direction):
        if len(direction.shape) == 4:
            res = np.zeros(shape=(direction.shape[0], self.input_dim[0], self.input_dim[1], 1))
            for k in range(direction.shape[0]):
                res[k,...,0] = np.flipud(np.reshape(np.matmul(np.transpose(self.m),
                                                            np.reshape(np.asarray(direction[k,...,0]), self.output_sq)),
                                                            [self.input_dim[0], self.input_dim[1]]))
        elif len(direction.shape) == 2:
            res = np.flipud(np.reshape(np.matmul(np.transpose(self.m), np.reshape(np.asarray(direction), self.output_sq)),
                                       [self.input_dim[0], self.input_dim[1]]))
        else:
            raise ValueError
        return res


    # Finds the inverse
    def inverse(self, y):
        if len(y.shape) == 4:
            res = np.zeros(shape=(y.shape[0], self.input_dim[0], self.input_dim[1], 1))
            for k in range(y.shape[0]):
                inp = np.reshape(np.asarray(y[k, ...,0]), self.output_sq)
                sol = np.linalg.solve(self.m, inp)
                res[k, ..., 0] = np.flipud(np.reshape(sol, [self.input_dim[0], self.input_dim[1]]))
        elif len(y.shape) == 2:
            inp = np.reshape(np.asarray(y), self.output_sq)
            sol = np.linalg.solve(self.m, inp)
            res = np.flipud(np.reshape(sol, [self.input_dim[0], self.input_dim[1]]))
        else:
            raise ValueError
        return res



# the exact PAT as numpy operator
class exact_PAT_operator(np_operator):
    def __init__(self, matrix_path, input_dim, output_dim):
        fData = h5py.File(matrix_path, 'r')
        inData = fData.get('A')
        rows = inData.shape[0]
        cols = inData.shape[1]
        print(rows, cols)
        self.m = np.matrix(inData)
        self.input_sq = input_dim[0]*input_dim[1]
        self.output_sq = output_dim[0]*output_dim[1]
        super(exact_PAT_operator, self).__init__(input_dim, output_dim)

    # computes the matrix multiplication with matrix
    def evaluate(self, y):
        y = np.flipud(np.asarray(y))
        if len(y.shape) == 4:
            res = np.zeros(shape=(y.shape[0], self.output_dim[0], self.output_dim[1], 1))
            for k in range(y.shape[0]):
                res[k,...,0] = np.reshape(np.matmul(self.m, np.reshape(y[k,...,0], self.input_sq)),
                                        [self.output_dim[0], self.output_dim[1]])
        elif len(y.shape) == 2:
            res = np.reshape(np.matmul(self.m, np.reshape(y, self.input_sq)), [self.output_dim[0], self.output_dim[1]])
        else:
            raise ValueError
        return res

    # matrix multiplication with the adjoint of the matrix
    def differentiate(self, point, direction):
        if len(direction.shape) == 4:
            res = np.zeros(shape=(direction.shape[0], self.input_dim[0], self.input_dim[1], 1))
            for k in range(direction.shape[0]):
                res[k,...,0] = np.flipud(np.reshape(np.matmul(np.transpose(self.m),
                                                            np.reshape(np.asarray(direction[k,...,0]), self.output_sq)),
                                                            [self.input_dim[0], self.input_dim[1]]))
        elif len(direction.shape) == 2:
            res = np.flipud(np.reshape(np.matmul(np.transpose(self.m), np.reshape(np.asarray(direction), self.output_sq)),
                                       [self.input_dim[0], self.input_dim[1]]))
        else:
            raise ValueError
        return res

# The model correction operator as numpy operator
class model_correction(np_operator):
    linear = False
    # categorizes experiments
    experiment_name = 'default_experiment'

    # makes sure the folders needed for saving the model and logging data are in place
    def generate_folders(self, path):
        paths = {}
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
            print(self.path)
            saver.restore(self.sess, tf.train.latest_checkpoint(self.path+'Data/'))
            print('Save restored')
        else:
            print('No save found')

    # clears computational graph
    def end(self):
        tf.reset_default_graph()
        self.sess.close()

    @abstractmethod
    def get_network(self, channels):
        pass

    @staticmethod
    # puts numpy array in form that can be fed into the graph
    def feedable_format(array):
        dim = len(array.shape)
        changed = False
        if dim == 2:
            array = np.expand_dims(np.expand_dims(array, axis=0), axis=-1)
            changed = True
        elif not dim == 4:
            raise ValueError
        return array, changed

    def __init__(self, path, data_sets):
        self.image_size = data_sets.train.image_resolution
        self.measurement_size = data_sets.train.y_resolution
        super(model_correction, self).__init__(self.measurement_size, self.measurement_size)
        self.data_sets = data_sets
        self.path = path+self.experiment_name+'/'
        self.generate_folders(self.path)
        self.UNet = self.get_network(channels=1)

        # start tensorflow sesssion
        self.sess = tf.InteractiveSession()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)


    @abstractmethod
    def train(self, learning_rate):
        pass

    @abstractmethod
    def log(self):
        pass


# Class that handels the datasets and training
class framework(object):
    # the data set name determines the saving folder and where to look for training data
    data_set_name = 'balls64'
    # the name of the matrix file to simulate the true forward operator
    matrix = 'threshSingleMatrix4Py.mat'
    # angular cut off
    angle = 60
    # tv parameter
    tv_param = 0.001

    def __init__(self, correction_model):
        self.correction_model = correction_model
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
        self.path = path_prefix + 'Saves/{}/'.format(self.data_set_name)

        # get the image and data space sizes
        self.image_size = self.data_sets.train.get_image_resolution()
        self.measurement_size = self.data_sets.train.get_y_resolution()

        # initializing the approximate PAT transform
        kgridBack = fpat.kgrid(data_path + 'kgrid_small.mat')
        kgridForw = fpat.kgrid(data_path + 'kgrid_smallForw.mat')
        plain_pat_operator = fpat.fastPAT(kgridBack, kgridForw, self.angle)
        self.appr_operator = approx_PAT_operator(plain_pat_operator, self.image_size, self.measurement_size)
        self.appr_odl = as_odl_operator(self.appr_operator)

        # initialize the correct PAT transform
        matrix_path = path_prefix+'Data/Matrices/' + self.matrix
        self.exact_operator = exact_PAT_operator(matrix_path, self.image_size, self.measurement_size)
        self.exact_odl = as_odl_operator(self.exact_operator)

        # initialize the correction operator
        self.cor_operator = self.correction_model(self.path, self.image_size,
                                                  self.measurement_size, self.appr_odl, self.data_sets)
        self.cor_odl = as_odl_operator(self.cor_operator)

    def train_correction(self, steps):
        for k in range(steps):
            self.cor_operator.train()
            if k%20 == 0:
                self.cor_operator.log()
        self.cor_operator.save()

    ####### TV reconstruction methods --- outdated!!!
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
        s = np.copy(start_point)
        x = space.element(s)

        # Run the optimization algoritm
        # odl.solvers.chambolle_pock_solver(x, functional, g, broad_op, tau = tau, sigma = sigma, niter=niter)
        odl.solvers.pdhg(x, functional, g, broad_op, tau=tau, sigma=sigma, niter=niter)
        return x




