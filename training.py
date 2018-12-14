from Adjoint_regularizition import Regularized
from Adjoint_network import TwoNets
from No_regularization import Unregularized
import Operators.Load_PAT2D_data as PATdata
import platform
from Framework import approx_PAT_matrix as ApproxPAT
from Framework import exact_PAT_operator as ExactPAT
import sys

if platform.node() == 'motel':
    prefix = '/local/scratch/public/sl767/ModelCorrection/'
else:
    prefix = ''

matrix_path = prefix+'Data/Matrices/threshSingleMatrix4Py.mat'
data_path = prefix+'Data/balls64/'
saves_path = prefix+'Saves/balls64/'

train_append = 'trainDataSet.mat'
test_append = 'testDataSet.mat'
data_sets = PATdata.read_data_sets(data_path + train_append, data_path + test_append)

input_dim = data_sets.train.image_resolution
output_dim = data_sets.train.y_resolution

approx = ApproxPAT(matrix_path=matrix_path, input_dim=input_dim, output_dim=output_dim)
exact = ExactPAT(matrix_path=matrix_path, input_dim=input_dim, output_dim=output_dim)

n = sys.argv[1]

if n == '1' or n == '0':
    correction = Regularized(path=saves_path, true_np=exact, appr_np=approx, data_sets=data_sets)

    rate = 5e-5
    for i in range(70):
        for k in range(200):
            correction.train(rate)
        correction.log()

    for i in range(70):
        for k in range(200):
            correction.train(rate / 10.0)
        correction.log()
    correction.end()

if n == '2' or n == '0':
    correction = TwoNets(path=saves_path, true_np=exact, appr_np=approx, data_sets=data_sets)

    rate = 5e-5
    for i in range(70):
        for k in range(200):
            correction.train(rate)
        correction.log()

    for i in range(70):
        for k in range(200):
            correction.train(rate / 10.0)
        correction.log()
    correction.end()

if n == '3' or n == '0':
    correction = Unregularized(path=saves_path, true_np=exact, appr_np=approx, data_sets=data_sets)

    rate = 5e-5
    for i in range(70):
        for k in range(200):
            correction.train(rate)
        correction.log()

    for i in range(70):
        for k in range(200):
            correction.train(rate / 10.0)
        correction.log()
    correction.end()

