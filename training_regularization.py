from Adjoint_regularizition import Regularized
import Operators.Load_PAT2D_data as PATdata
import platform
from Framework import approx_PAT_matrix as ApproxPAT
from Framework import exact_PAT_operator as ExactPAT

if platform.node() == 'motel':
    prefix = '/local/scratch/public/sl767/ModelCorrection/'
else:
    prefix = ''

matrix_path = prefix+'Data/Matrices/threshSingleMatrix4Py.mat'
data_path = prefix+'Data/balls64/'
saves_path = prefix+'Saves/balls64/'

print(saves_path)
print(data_path)

train_append = 'trainDataSet.mat'
test_append = 'testDataSet.mat'
data_sets = PATdata.read_data_sets(data_path + train_append, data_path + test_append)

INPUT_DIM = (64,64)
OUTPUT_DIM = (64,64)

approx = ApproxPAT(matrix_path=matrix_path, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
exact = ExactPAT(matrix_path=matrix_path, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)


TV = 0.001


if 0:
    correction = Regularized(path=saves_path, true_np=exact, appr_np=approx, lam=TV, data_sets=data_sets,
                             experiment_name='RegularizedAdjointOldNet')

    rate = 2e-4
    recursions = 1
    step_size = 0.2
    iterations = 50

    for i in range(iterations):
        print(f'Iteration {i+1}')
        for k in range(1000):
            correction.train(recursions, step_size, learning_rate=rate)
            if k % 50 == 0:
                correction.log(recursions, step_size)
        # recursions = recursions+1
    correction.save()
    #
    # correction.log_optimization(recursions=100, step_size=step_size, lam=0.0)
    # correction.log_gt_optimization(recursions=100, step_size=step_size, lam=0.0)
    # correction.log_approx_optimization(recursions=100, step_size=step_size, lam=0.0)
    #
    # correction.log_optimization(recursions=100, step_size=step_size, lam=TV)
    # correction.log_gt_optimization(recursions=100, step_size=step_size, lam=TV)
    # correction.log_approx_optimization(recursions=100, step_size=step_size, lam=TV)
    correction.end()


if 1:
    correction = Regularized(path=saves_path, true_np=exact, appr_np=approx, lam=TV, data_sets=data_sets,
                             experiment_name='RegularizedAdjointRekursiveOldNet')
    rate = 2e-4
    step_size = 0.2
    iterations = 60
    train_every_n = 10

    print('PreTraining')
    for k in range(3000):
        correction.train(1, step_size, learning_rate=rate, train_every_n=1)
    print('PreTraining Done')
    correction.save()

    i = 45
    recursion_list = []
    k = 1
    while k < i:
        recursions = (k + 1) ** 2
        recursion_list.append(recursions)
        k += 1


    if 1:
        while i <iterations:
            recursions = (i+1)**2
            recursion_list.append(recursions)
            print(f'Starting Iteration {i+1}, Max Recursions {recursions}')
            for k in range(1):
                for r in recursion_list:
                    train_every_n = int(r / 50) + 1
                    correction.train(recursions, step_size, learning_rate=rate, train_every_n=train_every_n)
                    if k % 20 == 0:
                        correction.log(recursions, step_size)
            if i >= 15 and i % 5 == 0:
                correction.save()
            i += 1

    # correction.log_optimization(recursions=100, step_size=step_size, lam=0.0)
    # correction.log_optimization(recursions=100, step_size=step_size, lam=TV)
    correction.end()



