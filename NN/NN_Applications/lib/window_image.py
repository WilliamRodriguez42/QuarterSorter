import numpy as np
import numpy as np

def get_window_shape(im_shape, input_shape, stride):
	return ((im_shape[0]-input_shape[0]) // stride[0] + 1, (im_shape[1]-input_shape[1]) // stride[1] + 1)

def window(im, input_shape, stride=1):
	window_shape = get_window_shape(im.shape, input_shape, stride)
	item_size = im.dtype.itemsize
	shape = (*window_shape, *input_shape)
	strides = (
		*(np.array(im.strides) * np.array(stride)).astype(np.int32),
		*im.strides
	)

	frames = np.lib.stride_tricks.as_strided(
		im,
		shape=shape,
		strides=strides,
		writeable=False)
	return frames

def test():
	import matplotlib.pyplot as plt
	import scipy.misc

	im = scipy.misc.ascent().astype(np.float32)
	stitched = np.zeros_like(im)

	input_shape = (73, 73)
	stride = (4, 4)

	result = window(im, input_shape, stride)

	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			si = i * stride[0]
			sj = j * stride[1]
			ei = si + input_shape[0]
			ej = sj + input_shape[1]
			stitched[si:ei, sj:ej] = result[i, j]

	plt.imshow(stitched.max() - stitched, cmap="Greys")
	plt.show()
