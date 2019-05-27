from Adjoint_regularizition import Regularized
from Adjoint_network import TwoNets
import Operators.Load_PAT2D_data as PATdata
import platform
from Framework import approx_PAT_matrix as ApproxPAT
from Framework import exact_PAT_operator as ExactPAT

if platform.node() == 'motel':
    prefix = '/local/scratch/public/sl767/ModelCorrection/'
else:
    prefix = ''

matrix_path = prefix+'Data/Matrices/threshSingleMatrix4Py.mat'
data_path = prefix+'Data/vessels/'
saves_path = prefix+'Saves/vessels/'

print(saves_path)
print(data_path)

train_append = 'vesselBatch2D_train.mat'
test_append = 'vesselBatch2D_test.mat'
data_sets = PATdata.read_data_sets(data_path + train_append, data_path + test_append, vessels=True)

INPUT_DIM = (64,64)
OUTPUT_DIM = (64,64)

approx = ApproxPAT(matrix_path=matrix_path, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
exact = ExactPAT(matrix_path=matrix_path, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)


TV = 0.001
step_size = 0.2
image = data_sets.test.default_batch(16)

def log_reference(model):
    model.log_gt_optimization(image, recursions=100, step_size=step_size, lam=0.0)
    model.log_approx_optimization(image, recursions=100, step_size=step_size, lam=0.0)
    model.log_gt_optimization(image, recursions=100, step_size=step_size, lam=TV)
    model.log_approx_optimization(image, recursions=100, step_size=step_size, lam=TV)


def log(model):
    model.log_optimization(image, recursions=100, step_size=step_size, lam=0.0)
    model.log_optimization(image, recursions=100, step_size=step_size, lam=TV)


correction = Regularized(path=saves_path, true_np=exact, appr_np=approx, lam=TV, data_sets=data_sets,
                         experiment_name='RegularizedAdjoint')
log_reference(correction)
log(correction)
correction.end()

correction = Regularized(path=saves_path, true_np=exact, appr_np=approx, lam=TV, data_sets=data_sets,
                         experiment_name='RegularizedAdjointRekursive')
log(correction)
correction.end()

correction = TwoNets(path=saves_path, true_np=exact, appr_np=approx, lam=TV, data_sets=data_sets,
                     experiment_name='TwoNets')
log(correction)
correction.end()

correction = TwoNets(path=saves_path, true_np=exact, appr_np=approx, lam=TV, data_sets=data_sets,
                     experiment_name='TwoNetsRekursive')
log(correction)
correction.end()

