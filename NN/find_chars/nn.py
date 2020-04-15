import time
from lib.vers import *
import numpy as np


start_time = time.time()
# =============================== Settings ===============================
batch_size = 100
num_epochs = 100
# num_batches = 60000 / 100 = (number of input elements) / (batch size)
input_shape = (73, 73)
input_size = np.prod(input_shape)

l0 = grumpy_layer(
	lr = 0.0006,
	influence = 0.0005,
	stable = 1e-10,
	refresh = 1e-6
	)

choice_scheme = list(np.arange(26).repeat(2))
#choice_scheme.extend([26] * 10)
output_scheme = np.zeros(26) #+ 10)
l0.set_scheme(
	input_size,
	np.array(choice_scheme, dtype=np.uint32), # How variations are distributed amongst choices
	np.array(output_scheme, dtype=np.uint32) # How choices are distributed amongst outputs
	)
# ========================================================================

split = 700
training_v = np.load('training_set/training_v.npy')[:split].reshape(-1, batch_size).astype(np.int32)
training_y = np.load('training_set/training_y.npy')[:split].reshape(-1, batch_size, l0.get_num_choices())
training_x = np.load('training_set/training_x.npy')[:split].reshape(-1, batch_size, l0.get_input_size())

testing_v = np.load('training_set/training_v.npy')[split:].astype(np.int32)
testing_y = np.load('training_set/training_y.npy')[split:]
testing_x = np.load('training_set/training_x.npy')[split:].reshape(-1, l0.get_input_size())

for j in range(num_epochs):
	dropout = training_x * (np.random.rand(*training_x.shape) * 0.5 + 0.5)

	print('Epoch ' + str(j))
	l0.refresh_lr()
	for i in range(training_x.shape[0]):
		feed(dropout[i], l0)
		back(
			dropout[i],
			training_y[i],
			l0,
			beta=0.995)

		if (i % 10 == 0):
			print(l0.get_avg())

		l0.update_lr()

	feed(testing_x, l0)
	back(
		testing_x,
		testing_y,
		l0,
		update=False)

	print('Testing average: ' + str(l0.get_avg()))
show_Z(l0, input_shape, 0)

stop_time = time.time()
print(stop_time - start_time)
