from PIL import Image
import glob
import numpy as np
import pickle
from numba import jit
import time

start_time = time.time()


def parse():
	files = glob.glob('dataset/*')
	files.sort()
	files.sort(key=len)

	output = np.zeros((len(files) // 2, 2, 1200, 1600), dtype=np.uint8)

	out_i = 0
	for file in files:
		print(file)
		im_mat = np.array(Image.open(file).convert('L'))
		if '.b.jpg' in file:
			output[out_i // 2, 1] = im_mat
		else:
			output[out_i // 2, 0] = im_mat

		out_i += 1

	pickle.dump(output, open('pickle/dataset.pkl', 'wb+'))

parse()

stop_time = time.time()
print(stop_time - start_time)
