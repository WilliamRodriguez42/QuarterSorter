#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
import time

cpdef np.ndarray window(im, (int, int) input_shape, (int, int) stride, order=None):
	if order is None:
		order = get_window_indices(im.shape, input_shape, stride)

	return im.flatten()[order.flatten()].reshape((order.shape[0], order.shape[1], input_shape[0], input_shape[1]))

cpdef np.ndarray get_window_indices((int, int) im_shape, (int, int) input_shape, (int, int) stride):
	cdef (int, int) window_shape = get_window_shape(im_shape, input_shape, stride)
	cdef int num_windows = np.prod(window_shape)

	cdef np.ndarray template = np.empty(input_shape, dtype=np.uint32)
	cdef int j
	for j in range(input_shape[1]):
		template[:, j] = np.arange(input_shape[0])*im_shape[1]+j
	template = template.flatten()

	print("Using {} bytes".format(num_windows * template.size * 4))
	cdef np.ndarray order = np.repeat(
								np.repeat(
									np.arange(window_shape[0], dtype=np.uint32)*im_shape[1]*stride[0],
									window_shape[1]
								) + np.tile(
									np.arange(window_shape[1], dtype=np.uint32)*stride[1],
									window_shape[0]
								),
								template.size
							) + np.tile(
								template,
								num_windows
							)

	return order.reshape((window_shape[0], window_shape[1], -1))

cpdef (int, int) get_window_shape((int, int) im_shape, (int, int) input_shape, (int, int) stride):
	return ((im_shape[0]-input_shape[0]) // stride[0] + 1, (im_shape[1]-input_shape[1]) // stride[1] + 1)

def profile(shape, input_shape, stride, iterations = 100):
	im = np.random.rand(*shape)

	avg_time = 0

	start_time = time.time()
	order = get_window_indices(im.shape, input_shape, stride)
	stop_time = time.time()
	print("Getting the order took {} seconds".format(stop_time - start_time))

	for i in range(iterations):

		start_time = time.time()
		window(im, input_shape, stride, order)
		stop_time = time.time()

		avg_time += float(stop_time - start_time) / iterations

	print(avg_time)

cpdef test():
	import matplotlib.pyplot as plt
	import scipy.misc

	cdef np.ndarray im = scipy.misc.ascent().astype(np.float32)
	cdef np.ndarray stitched = np.zeros_like(im)

	cdef (int, int) input_shape = (73, 73)
	cdef (int, int) stride = (4, 4)

	result = window(im, input_shape, stride)

	cdef int i, j
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			si = i * stride[0]
			sj = j * stride[1]
			ei = si + input_shape[0]
			ej = sj + input_shape[1]
			stitched[si:ei, sj:ej] = result[i, j]

	plt.imshow(stitched.max() - stitched, cmap="Greys")
	plt.show()
