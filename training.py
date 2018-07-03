import Framework as fr

frame = fr.framework()
batch_size = 64
learning_rate = 0.0001
for k in range(20):
    frame.train_correction(2000, batch_size=batch_size, learning_rate=learning_rate)