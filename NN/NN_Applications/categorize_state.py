import lib.window_image as wi
import lib.coin_center as cc
import create_state_kernels as csk
from lib.vers import *
from PIL import Image

import pickle
import numpy as np
import time
import sys
import io

def categorize(b, top=True):
	image = Image.open(io.BytesIO(b)).convert('L')
	arr = np.array(image, dtype=np.float32)

	if top:
		unwrapped = cc.parse_top_coin(arr, extended=False)
	else:
		unwrapped = cc.parse_bottom_coin(arr, extended=False)
	unwrapped = unwrapped[:-lob_off, :]

	repeated = np.zeros((unwrapped.shape[0], unwrapped.shape[1]+input_shape[1]-1), dtype=np.float32)
	repeated[:, :unwrapped.shape[1]] = unwrapped
	repeated[:, unwrapped.shape[1]:] = repeated[:, :repeated.shape[1]-unwrapped.shape[1]]

	windowed = wi.window(repeated, input_shape, stride)
	window_shape = windowed.shape[:2]

	windowed = windowed.reshape(-1, input_size)

	l0.feed(windowed)
	top_predictions = l0.get_oi().reshape(window_shape)

	max_pool_count = np.zeros((window_shape[1] // j_pool + 1, 26))

	for i in range(window_shape[0]):
		for j in range(window_shape[1]):
			pred = top_predictions[i, j]

			if pred != 26:
				mj = j // j_pool

				max_pool_count[mj, pred] += 1

	max_pool = np.argmax(max_pool_count, axis=1)

	repeated = np.zeros((max_pool_count.shape[0]+state_kernel_shape[0]-1, max_pool_count.shape[1]), dtype=np.float32)
	repeated[:max_pool_count.shape[0], :] = max_pool_count
	repeated[max_pool_count.shape[0]:, :] = repeated[:repeated.shape[0]-max_pool_count.shape[0], :]

	max_pool_windows = wi.window(repeated, state_kernel_shape, (1, 1)).reshape(-1, state_kernel_size)
	max_pool_windows = max_pool_windows / (np.sqrt(np.sum(np.square(max_pool_windows), axis=1)) + 1e-9)[:, None]
	max_pool_windows = max_pool_windows / (np.sqrt(np.sum(np.square(max_pool_windows), axis=0)) + 1e-9)
	output = max_pool_windows.dot(state_kernels)
	pos = np.unravel_index(np.argmax(output), output.shape)[::-1]

	return (pos[0], output.max(), csk.states[pos[0]])







# ================================================= Settings =====================================================
import matplotlib.pyplot as plt

l0 = pickle.load(open('l0.pkl', 'rb'))
to_prune = np.array([
	       1,            3,
	5,     6,   7,       8,
	      11,  12,  13,  14,
	15,   16,       18,  19,

	25,   26,       28,  29,
	      31,  32,  33,  34,
	35,   36,  37,  38,

	45,   46,  47,  48,  49,
	50,   51,       53,
	55,        57,  58,
	      61,  62,       64,
	
	      71,       73,
	75,   76,  77,       79,
	80,   81,  82,  83,  84,
	85,   86,  87,  88,
	      91,
	           97,  98,  99,
	100,      102, 103, 104,
	105, 106, 107,      109,
	110, 111, 112,      114,
	115, 116, 117,      119,
	     121, 122, 123, 124,
	125, 126, 127, 128, 129,])
l0.prune(to_prune)

lob_off = 50
i_pool = 30
j_pool = 6

input_shape = (73, 73)
input_size = np.prod(input_shape)
stride = (8, 8)

SK = csk.create_state_kernels(20)
state_kernel_shape = SK.shape[1:]
state_kernel_size = np.prod(state_kernel_shape)
state_kernels = SK.reshape(-1, state_kernel_size).T
SK = None

# ================================================================================================================



if __name__ == '__main__':
	file = open('../NN_Shrunk/dataset/7.t.jpg', 'rb')
	content = file.read()
	file.close()

	categorize(content, top=True)

