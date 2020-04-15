import lib.coin_center as cc
from id_stor.manage_id_stor import read_id_stor, create_id_stor
import matplotlib.pyplot as plt
import numpy as np
import lib.window_image as wi
import sys
import time

start_time = time.time()
id_stor = create_id_stor(cc.dataset.shape[0])
read_id_stor(id_stor)
stop_time = time.time()
print(stop_time - start_time)

num_ids = 0
image_width = 1600
for ids_arr in id_stor:
	for ids in ids_arr:
		for id in ids:
			num_ids += 1
stop_time = time.time()
print(stop_time - start_time)

image_width = 1600
top_unwrapped, _ = cc.parse_dataset_element(0)
image_scale = top_unwrapped.shape[1] / image_width
rect_width = int(30 * image_scale + 0.5)
rect_height = int(30 * image_scale + 0.5)
stop_time = time.time()
print(stop_time - start_time)

stride = (1, 1)

temp, _ = cc.parse_dataset_element(0)
window_shape = wi.get_window_shape(temp.shape, (rect_height, rect_width), stride)
num_windows = np.prod(window_shape)
stop_time = time.time()
print(stop_time - start_time)

training_x = np.zeros((num_ids * num_windows, rect_width, rect_height), dtype=np.float32)
training_v = np.zeros(num_ids * num_windows, dtype=np.uint8)
training_y = np.zeros((num_ids * num_windows, 26), dtype=np.uint8)
stop_time = time.time()
print(stop_time - start_time)

curr_id = 0
for i, ids_arr in enumerate(id_stor):
	top_unwrapped, bot_unwrapped = cc.parse_dataset_element(i)

	top_unwrapped = top_unwrapped - top_unwrapped.min()
	top_unwrapped = top_unwrapped / top_unwrapped.max()
	bot_unwrapped = bot_unwrapped - bot_unwrapped.min()
	bot_unwrapped = bot_unwrapped / bot_unwrapped.max()

	start_time = time.time()
	top_windowed = wi.window(top_unwrapped, (rect_height, rect_width), stride)
	top_windowed_v = np.ones(num_windows, dtype=np.int32) * 27
	bot_windowed = wi.window(bot_unwrapped, (rect_height, rect_width), stride)
	bot_windowed_v = np.ones(num_windows, dtype=np.int32) * 27
	stop_time = time.time()
	print(stop_time - start_time)

	sys.stdout.write("Working on {} of {}\r".format(i, num_ids * num_windows))
	sys.stdout.flush()

	for j, ids in enumerate(ids_arr):
		for ((x, y), c, _) in ids:
			sj = int(x * image_scale + 0.5)
			si = int(y * image_scale + 0.5)
			ej = si + rect_width
			ei = si + rect_height

			# if j == 0:
			# 	training_x[curr_id] = top_unwrapped[si:ei, sj:ej]
			# else:
			# 	training_x[curr_id] = bot_unwrapped[si:ei, sj:ej]

			c_index = ord(c) - ord('a')
			training_v[curr_id] = c_index
			training_y[curr_id, c_index] = 1
			curr_id += 1

np.save('training_set/training_x', training_x)
np.save('training_set/training_v', training_v)
np.save('training_set/training_y', training_y)
