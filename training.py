import Framework as fr
import numpy as np
import matplotlib.pyplot as plt

frame = fr.framework()
batch_size = 64
learning_rate = 0.0001
# for k in range(20):
#     frame.train_correction(2000, batch_size=batch_size, learning_rate=learning_rate)

###
# list = []
# for k in range(10):
#     list.append(3**(k-7))
# frame.find_tv_param(list)

### consistency test
appr, true, image = frame.data_sets.train.next_batch(1)
appr2 = frame.pat_operator.evaluate(image)
print(np.sum(np.square(appr-appr2)))
plt.figure(0)
plt.imshow(appr[0,...])
plt.show()
plt.figure(1)
plt.imshow(appr2[0,...])
plt.show()