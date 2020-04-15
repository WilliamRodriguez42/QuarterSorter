import lib.window_image as wi
import lib.coin_center as cc
import create_state_kernels as csk
from lib.vers import *

import pygame
import pickle
import numpy as np
import time
import sys

def load_dataset_element(k):
	start_time = time.time()
	top_unwrapped, bot_unwrapped = cc.parse_dataset_element(k//2, extended=False)
	if k % 2 == 0:
		unwrapped = top_unwrapped[:-lob_off, :]
	else:
		unwrapped = bot_unwrapped[:-lob_off, :]

	repeated = np.zeros((unwrapped_shape[0], unwrapped_shape[1]+input_shape[1]-1), dtype=np.float32)
	repeated[:, :unwrapped_shape[1]] = unwrapped
	repeated[:, unwrapped_shape[1]:] = repeated[:, :repeated.shape[1]-unwrapped_shape[1]]

	windowed = wi.window(repeated, input_shape, stride)
	window_shape = windowed.shape[:2]

	windowed = windowed.reshape(-1, input_size)

	l0.feed(windowed, dtype=np.float64)
	top_predictions = l0.get_oi().reshape(window_shape)

	max_pool_count = np.zeros((window_shape[1] // j_pool + 1, 26))

	for i in range(window_shape[0]):
		for j in range(window_shape[1]):
			pred = top_predictions[i, j]

			if pred != 26:
				mj = j // j_pool

				max_pool_count[mj, pred] += 1

	max_pool = np.argmax(max_pool_count, axis=1)
	max_pool_mask = max_pool_count[np.arange(max_pool.shape[0]), max_pool] > 0

	repeated = np.zeros((max_pool_count.shape[0]+state_kernel_shape[0]-1, max_pool_count.shape[1]), dtype=np.float32)
	repeated[:max_pool_count.shape[0], :] = max_pool_count
	repeated[max_pool_count.shape[0]:, :] = repeated[:repeated.shape[0]-max_pool_count.shape[0], :]

	max_pool_windows = wi.window(repeated, state_kernel_shape, (1, 1)).reshape(-1, state_kernel_size)
	max_pool_windows = max_pool_windows / (np.sqrt(np.sum(np.square(max_pool_windows), axis=0)) + 1e-9)
	output = max_pool_windows.dot(state_kernels)
	pos = np.unravel_index(np.argmax(output), output.shape)[::-1]

	if output.max() > 1:
		print(csk.states[pos[0]])
		print(output.max())
	stop_time = time.time()
	#print(stop_time - start_time)







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
l0.astype(np.float64)

lob_off = 50
i_pool = 30
j_pool = 6

input_shape = (73, 73)
input_size = np.prod(input_shape)
stride = (8, 8)
top_unwrapped, _ = cc.parse_dataset_element(0, extended=False)
unwrapped = top_unwrapped[:-lob_off, :]
unwrapped_shape = unwrapped.shape
unwrapped = None
top_unwrapped = None

SK = csk.create_state_kernels(20)
state_kernel_shape = SK.shape[1:]
state_kernel_size = np.prod(state_kernel_shape)
state_kernels = SK.reshape(-1, state_kernel_size).T
SK = None

# ================================================================================================================




for k in range(cc.dataset.shape[0]):
	load_dataset_element(k)