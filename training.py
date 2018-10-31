import Framework as fr
import numpy as np
import matplotlib.pyplot as plt

frame = fr.framework_regularised()
batch_size = 64
learning_rate = 0.001
frame.train_correction(2, batch_size=batch_size, learning_rate=learning_rate)
print('Test Training Successful')
for k in range(20):
    rate = learning_rate/(k+3)
    frame.train_correction(2000, batch_size=batch_size, learning_rate=rate)

###
# list = []
# for k in range(10):
#     list.append(3**(k-7))
# frame.find_tv_param(list)
