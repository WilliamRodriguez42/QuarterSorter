import numpy as np
import timeit

N = 1600
def func():
	im = np.random.rand(N, N)
	input_shape = (2, 2)
	stride = (1, 1)

	im_flattened = im.reshape(-1)

	window_shape = ((im.shape[0]-input_shape[0]) // stride[0] + 1, (im.shape[1]-input_shape[1]) // stride[1] + 1)
	window_size = np.prod(window_shape)

	template = np.zeros(input_shape, dtype=np.int32)
	for j in range(input_shape[1]):
		template[:, j] = np.arange(input_shape[0])*im.shape[1]+j
	template = template.reshape(-1)

	order = np.empty((window_size, template.size))
	i_iter = (np.arange(window_shape[0])*im.shape[1]*stride[0]).repeat(window_shape[1])
	j_iter = np.tile(np.arange(window_shape[1])*stride[1], window_shape[0])
	to_add = (i_iter + j_iter).repeat(template.size)
	order = np.tile(template, window_size)
	order = order + to_add

	#print(im_flattened[order].reshape(window_size, input_shape[0], input_shape[1]))

number = 100
print(timeit.timeit(func, number=number) / number)
