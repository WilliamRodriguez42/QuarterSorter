#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from scipy import signal#, ndimage
from PIL import Image
from libc cimport math
import pickle
from threading import Thread

class ThreadWithReturnValue(Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
		Thread.__init__(self, group, target, name, args, kwargs)
		self._return = None

	def run(self):
		if self._target is not None:
			self._return = self._target(*self._args, **self._kwargs)

	def join(self, *args):
		Thread.join(self, *args)
		return self._return

def save(mat, name):
	mat = mat - mat.min()
	Image.fromarray((mat / (mat.max() + 1e-10) * 255).astype(np.uint8)).save('images/' + name)

cpdef (np.ndarray) imresize2(np.ndarray im, int r, int c):
	cdef int i, j, ui, uj
	cdef float pi, pj

	cdef np.ndarray result = np.zeros((r, c), dtype=np.float32)
	cdef float[:, :] result_view = result
	cdef float[:, :] im_view = im

	with nogil:
		for i in range(r):
			for j in range(c):
				pi = float(i) / (r - 1)
				pj = float(j) / (c - 1)
				ui = int(pi * (im.shape[0] - 1) + 0.5)
				uj = int(pj * (im.shape[1] - 1) + 0.5)

				result_view[i, j] = im_view[ui, uj]

	return result

cpdef (np.ndarray) imresize(np.ndarray im, float a):
	#cdef (int, int) shape = (int(im.shape[1] * a), int(im.shape[0] * a))
	#return np.array(Image.fromarray(im.astype(np.float32)).resize(shape))
	cdef int r = int(im.shape[0] * a)
	cdef int c = int(im.shape[1] * a)

	return imresize2(im, r, c)


cpdef (np.ndarray) create_top_kernel():

	# Calculate where on the scaled image the edge of the coin should lie
	cdef int w = int(top_coin_radius * 2 * a) + 2
	cdef float urad = w / 2.0
	cdef float mrad = urad - 1
	cdef float lrad = mrad - 1
	cdef int hw = w / 2

	# Create a new kernel for finding the coin edge
	cdef np.ndarray top_kernel = np.zeros((w, w), dtype=np.float32)
	cdef float ci = top_kernel.shape[0] / 2.0
	cdef float cj = top_kernel.shape[1] / 2.0
	cdef float[:, :] top_kernel_view = top_kernel

	cdef float di, dj, r

	cdef int i, j
	for i in range(hw):
		for j in range(hw):
			di = i - ci
			dj = j - cj
			r = math.sqrt(di**2 + dj**2)
			if r < mrad and r >= lrad:
				top_kernel_view[i, j] = 1
			elif r >= mrad and r <= urad:
				top_kernel_view[i, j] = -1

	top_kernel[hw:, hw:] = np.flip(top_kernel[:hw, :hw], axis=(0, 1))
	top_kernel[:hw, hw:] = np.flip(top_kernel[:hw, :hw], axis=1)
	top_kernel[hw:, :hw] = np.flip(top_kernel[:hw, :hw], axis=0)

	save(top_kernel, 'top_kernel.png')
	return top_kernel


cpdef (float, float) find_center(np.ndarray image, np.ndarray kernel):
	# Scale down
	cdef np.ndarray scaled = imresize(image, a)
	scaled = scaled / scaled.max()

	# Convolve with filtered-scaled coin image
	#output = ndimage.convolve(scaled, kernel)
	cdef np.ndarray output = signal.correlate(scaled, kernel, mode='valid', method='fft').astype(np.float32)
	cdef float[:, :] output_view = output

	# Calculate the average of these predictions
	cdef float center_i = 0
	cdef float center_j = 0
	cdef float c = 0
	cdef float limit = output.max() * 0.5

	cdef int i, j
	with nogil:
		for i in range(output.shape[0]):
			for j in range(output.shape[1]):
				if output_view[i, j] > limit:
					center_i += i * output_view[i, j]
					center_j += j * output_view[i, j]
					c += output_view[i, j]

	center_i = (center_i / c + kernel.shape[0] / 2) / a
	center_j = (center_j / c + kernel.shape[1] / 2) / a

	return (center_i, center_j)

cpdef show_image_center(np.ndarray im, float center_i, float center_j, name=None):
	im = im.astype(np.float32)
	cdef float[:, :] im_view = im

	cdef float di, dj, r
	cdef int i, j
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			di = center_i - i
			dj = center_j - j
			r = math.sqrt(di**2 + dj**2)

			if bot_coin_radius-10 < r and r < bot_coin_radius+10:
				im_view[i, j] += 200

	"""cdef int pad_size = 500
	cdef np.ndarray padded_im = np.pad(im, pad_size, mode='mean')
	cdef np.ndarray top_kernel_scaled = imresize(top_kernel, 1/a) * 100

	i = int(center_i - top_kernel_scaled.shape[0] / 2 + 0.5) + pad_size
	j = int(center_j - top_kernel_scaled.shape[1] / 2 + 0.5) + pad_size

	padded_im[i:i+top_kernel_scaled.shape[0], j:j+top_kernel_scaled.shape[1]] += top_kernel_scaled"""
	if name != None:
		save(im, name)
	else:
		return im

cpdef (float, float) find_bottom_center():
	cdef np.ndarray bot = pickle.load(open('pickle/bot_center.pkl', 'rb'))
	return (bot[0], bot[1])

cpdef (float, float) find_top_center():
	cdef np.ndarray top = pickle.load(open('pickle/top_center.pkl', 'rb'))
	return (top[0], top[1])

# Find zero position of coins
def create_zero_pos():
	top_i, top_j = find_top_center()
	show_image_center(dataset[0, 0], top_i, top_j, 'top_zero_pos.png')
	bot_i, bot_j = find_bottom_center()
	show_image_center(dataset[0, 1], bot_i, bot_j, 'bottom_zero_pos.png')

	zero_pos = np.array([[top_i, top_j], [bot_i, bot_j]], dtype=np.float32)
	#pickle.dump(zero_pos, open('pickle/zero_pos.pkl', 'wb+'))
	return zero_pos


cpdef (np.ndarray) parse_coin(np.ndarray data, float center_i, float center_j, float coin_radius, float height_scale):
	# Get matrix
	data = data / data.max() # Between [0, 1]
	cdef float[:, :] data_view = data

	# Unwrap the coin using the calculated coin center
	cdef int width = int(top_coin_radius * 2 * math.pi)
	cdef int height = 160
	cdef np.ndarray unwrapped = np.zeros((height, width), dtype=np.float32)
	cdef float[:, :] unwrapped_view = unwrapped

	cdef float an, rad, dj, di
	cdef int ui, uj

	cdef int i, j
	with nogil:
		for i in range(height):
			for j in range(width):
				an = j / float(width) * 2*math.pi
				rad = coin_radius - i*height_scale

				dj = rad * math.cos(an)
				di = rad * math.sin(an)

				ui = int(center_i+di+0.5)
				uj = int(center_j+dj+0.5)
				unwrapped_view[i, j] = data_view[ui, uj]

	cdef np.ndarray repeated = np.zeros((height, int(width*1.5)), dtype=np.float32)
	repeated[:, :width] = unwrapped
	repeated[:, width:] = repeated[:, :repeated.shape[1]-width]

	return repeated

cpdef np.ndarray parse_top_coin(np.ndarray top, float top_i, float top_j):
	cdef np.ndarray unwrapped = parse_coin(top, top_i, top_j, top_coin_radius, 1)
	return unwrapped

cpdef np.ndarray parse_bottom_coin(np.ndarray bot, float bot_i, float bot_j):
	cdef np.ndarray unwrapped = parse_coin(bot, bot_i, bot_j, bot_coin_radius, bt_ratio)
	return unwrapped

cpdef parse_top_bottom(np.ndarray top, np.ndarray bot):
	# Crop the images
	top = top[crop_top:crop_bottom, crop_left:crop_right]
	bot = bot[crop_top:crop_bottom, crop_left:crop_right]

	# Parse the text from the top image
	#top_ij = find_top_center(top)
	#cdef float top_i = top_ij[0]
	#cdef float top_j = top_ij[1]

	cdef float top_i = zero_pos_view[0, 0]
	cdef float top_j = zero_pos_view[0, 1]
	cdef float bot_i = zero_pos_view[1, 0]
	cdef float bot_j = zero_pos_view[1, 1]


	top_thread = ThreadWithReturnValue(target=parse_top_coin, args=[top, top_i, top_j])
	top_thread.start()

	# Move the bottom image based on the results of the top image
	#cdef float di = top_i - zero_pos_view[0, 0]
	#cdef float dj = top_j - zero_pos_view[0, 1]
	#cdef float bot_i = zero_pos_view[1, 0] + di * bt_ratio
	#cdef float bot_j = zero_pos_view[1, 1] - dj * bt_ratio

	# Parse the text from the bottom image
	cdef np.ndarray bot_unwrapped = parse_bottom_coin(bot, bot_i, bot_j)
	#show_image_center(bot, bot_i, bot_j, 'bottom_centered.png')

	cdef np.ndarray top_unwrapped = top_thread.join()
	#show_image_center(top, top_i, top_j, 'top_centered.png')

	return (top_unwrapped, bot_unwrapped)

cpdef crop_test():
	cdef np.ndarray top = dataset[0, 0]
	cdef np.ndarray bot = dataset[0, 1]

	top = top[crop_top:crop_bottom, crop_left:crop_right]
	bot = bot[crop_top:crop_bottom, crop_left:crop_right]

	save(top, 'top_cropped.png')
	save(bot, 'bot_cropped.png')

# Maximum cropping parameters
cdef int crop_left = 250
cdef int crop_right = 1600 - 400
cdef int crop_top = 150
cdef int crop_bottom = 1200 - 150

# Set the coin radius
cdef float top_coin_radius = 415
cdef float bot_coin_radius = 380

# Set up a ratio to convert top coin pixels to bottom coin pixels
cdef float bt_ratio = bot_coin_radius / top_coin_radius

# Scaling factor for convolution
cdef float a = 0.2

# Load the data
dataset = pickle.load(open('pickle/dataset.pkl', 'rb')).astype(np.float32)
#cdef np.ndarray zero_pos = pickle.load(open('pickle/zero_pos.pkl', 'rb')).astype(np.float32)

# Create kernels
#cdef np.ndarray top_kernel = create_top_kernel()
cdef np.ndarray zero_pos = create_zero_pos()
cdef float[:, :] zero_pos_view = zero_pos

# Zeroing was done on un-cropped images, offset for cropped images
zero_pos_view[0, 0] -= crop_top
zero_pos_view[1, 0] -= crop_top
zero_pos_view[0, 1] -= crop_left
zero_pos_view[1, 1] -= crop_left

cpdef parse_dataset_element(int i):
	top_unwrapped, bot_unwrapped = parse_top_bottom(dataset[i, 0], dataset[i, 1])

	return top_unwrapped, bot_unwrapped
