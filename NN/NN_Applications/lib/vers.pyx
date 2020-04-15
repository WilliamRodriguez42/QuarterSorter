import numpy as np
cimport numpy as np
from math import ceil, floor
from PIL import Image
import sys

import random
import matplotlib.pyplot as plt
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

random.seed(0)
np.random.seed(0)

def imsave(mat, name):
	mat = mat.astype(np.float64)
	mat = mat - mat.min()
	mat = mat / mat.max()
	mat = mat * 255

	new_im = Image.fromarray(mat).convert('L')
	new_im.save(name)

def imshow(image, file_name=None):
	plt.tick_params(
		axis='both',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		left=False,
		right=False,
		labelbottom=False,
		labeltop=False,
		labelleft=False,
		labelright=False
		) # labels along the bottom edge are off
	plt.imshow(image.max()-image, cmap='Greys')
	if file_name:
		imsave(image, file_name)

cpdef np.ndarray sub_dot(np.ndarray x, np.ndarray Z):
	return x.dot(Z) / x.shape[1]

cdef class grumpy_layer:
	cdef double lr
	cdef double influence
	cdef double stable
	cdef double refresh
	cdef np.ndarray Z
	cdef np.ndarray sqr
	cdef np.ndarray vv
	cdef np.ndarray choice_scheme
	cdef np.ndarray output_scheme
	cdef double avg
	cdef int num_variations
	cdef int num_choices
	cdef int num_outputs
	cdef int input_size
	cdef np.ndarray _v
	cdef np.ndarray _ci
	cdef np.ndarray _c
	cdef np.ndarray _oi
	cdef np.ndarray _o
	cdef np.ndarray _oci
	cdef np.ndarray _one_oc
	cdef np.ndarray ideal_x

	cpdef show_Z(grumpy_layer l, input_shape, output_index):
		# Create a list of variations for this output
		vo = []

		# Get choices for this output
		choice_mask = (l.output_scheme == output_index)

		# Normalize Z
		Z = l.Z - l.Z.min()
		Z = Z / Z.max()

		# Get variations for each choice in this output
		max_variations_for_choice = 0
		for i in range(l.num_choices):
			if choice_mask[i]:
				# Create a list of variations for this choice
				vc = []

				variation_mask = (l.choice_scheme == i)
				variations = Z[:, variation_mask]

				num_variations_for_choice = variation_mask.sum()
				if num_variations_for_choice > max_variations_for_choice:
					max_variations_for_choice = num_variations_for_choice

				for j in range(num_variations_for_choice):
					# Reshape variation into image
					vc.append(variations[:, j].reshape(input_shape))

				vo.append(vc)

		# Create a matrix to hold all the variations
		sub_im_width = input_shape[1]+10
		sub_im_height = input_shape[0]+10
		image = np.zeros((len(vo)*sub_im_height, max_variations_for_choice*sub_im_height))
		for i in range(len(vo)):
			variations = vo[i]
			for j in range(len(variations)):
				si = i*sub_im_height + 5
				ei = si + input_shape[0]
				sj = j*sub_im_width + 5
				ej = sj + input_shape[1]
				image[si:ei, sj:ej] = variations[j]

		imsave(image, 'Z.png')

	cpdef feed(grumpy_layer l, np.ndarray x, dtype=np.float32):
		cdef int batch_size = x.shape[0]
		cdef np.ndarray batch_iterator = np.arange(batch_size)
		cdef int i

		# Get value of each variation
		cdef np.ndarray _v = x.dot(l.Z) / x.shape[1] # (batch_size, num_variations)
		
		"""cdef int num_cores = 4
		cdef int seg_size = l.Z.shape[1] // 4 + 1
		cdef np.ndarray _v = np.zeros((x.shape[0], l.Z.shape[1]))

		thread_pool = []
		for i in range(num_cores):
			s = slice(i*seg_size, (i+1)*seg_size)
			thread = ThreadWithReturnValue(target=sub_dot, args=[x, l.Z[:, s]])
			thread.start()
			thread_pool.append(thread)

		for i in range(num_cores):
			s = slice(i*seg_size, (i+1)*seg_size)
			_v[:, s] = thread_pool[i].join()"""

		# Index of the best variation for each choice
		cdef np.ndarray _ci = np.zeros((batch_size, l.num_choices), dtype=np.uint32) # (batch_size, num_choices)

		# Value of the best variation for each choice
		cdef np.ndarray _c = np.zeros((batch_size, l.num_choices), dtype=dtype) # (batch_size, num_choices)

		cdef np.ndarray variation_mask

		for i in range(l.num_choices):
			variation_mask = (l.choice_scheme == i)
			_ci[:, i] = np.argmax(_v * variation_mask, axis=1)
			_c[:, i] = _v[batch_iterator, _ci[:, i]]

		# Index of the best choice for an output
		cdef np.ndarray _oi = np.zeros((batch_size, l.num_outputs), dtype=np.uint32) # (batch_size, num_outputs)

		# Value of best variation for output
		cdef np.ndarray _o = np.zeros((batch_size, l.num_outputs), dtype=dtype)

		# Index of best variation for any output and choice
		cdef np.ndarray _oci = np.zeros((batch_size, l.num_outputs), dtype=np.uint32)

		# One hot representation of the best variation for any output and choice
		cdef np.ndarray _one_oc = np.zeros((batch_size, l.num_variations), dtype=np.int8) # (batch_size, num_variations)

		cdef np.ndarray choice_mask

		for i in range(l.num_outputs):
			choice_mask = (l.output_scheme == i)
			_oi[:, i] = np.argmax(_c * choice_mask, axis=1)
			_o[:, i] = _c[batch_iterator, _oi[:, i]]

			_oci[:, i] = _ci[batch_iterator, _oi[:, i]]
			_one_oc[batch_iterator, _oci[:, i]] = 1

		l._v = _v
		l._ci = _ci
		l._c = _c
		l._oi = _oi
		l._o = _o
		l._oci = _oci
		l._one_oc = _one_oc

	cpdef back(grumpy_layer l, np.ndarray x, np.ndarray one_c, beta_vv=0, beta_sqr=0, double random_selection_rate = 0.0, int update=True):
		# Need to get one_oc (one hot best choice for each output)
		cdef int batch_size = x.shape[0]
		cdef np.ndarray batch_iterator = np.arange(batch_size)
		cdef int i, j

		cdef np.ndarray one_oc = np.zeros((batch_size, l.num_variations), dtype=np.int8)
		cdef np.ndarray incorrect = np.zeros((batch_size, l.num_outputs), dtype=np.int8)
		cdef np.ndarray variation_mask = np.zeros((batch_size, l.num_variations), dtype=bool)
		cdef np.ndarray choice_mask, choices_for_output, choice_index, variations_for_best_choice, best_variation_index_for_choice, worst_variation_index_for_choice

		for i in range(l.num_outputs):
			# Get best choice for this output
			choice_mask = (l.output_scheme == i)						# What choices belong to this output
			choices_for_output = one_c * choice_mask					# Find the one hot representation for this set of choices
			choice_index = np.argmax(choices_for_output, axis=1)		# Find the index of the best choice for each batch
			incorrect[:, i] = np.not_equal(choice_index, l._oi[:, i])	# Find the total number of incorrect guesses for this output

			index_of_variation_to_correct = np.zeros(batch_size, dtype=np.uint32)
			
			for j in range(batch_size):
				# Find minimum non-zero variation
				if incorrect[j, i]:
					variation_mask[j] = l.choice_scheme == choice_index[j]

					n = np.nonzero(variation_mask[j])[0]
					
					# Sometimes don't support the best variation, but rather a random one
					do_support_rando = random.random()
					if do_support_rando > 1 - random_selection_rate:
						# Choose a random variation to support
						k = int(random.random()*n.size)
						m = n[k]
						index_of_variation_to_correct[j] = m

					else:
						k = np.argmax(l._v[j, n])
						m = n[k]
						index_of_variation_to_correct[j] = m

			one_oc[batch_iterator, index_of_variation_to_correct] = 1

			#variations_for_best_choice = (l._v * variation_mask)
			#index_of_variation_to_correct = np.argmax(variations_for_best_choice, axis=1)
			#one_oc[batch_iterator, index_of_variation_to_correct] = 1

		# Find the average
		l.avg = 1 - np.mean(incorrect, axis=0)

		cdef np.ndarray dz_optimized
		if update:
			# Find the changes we need to make
			dv = (one_oc - l._one_oc) * incorrect
			dz = x.T.dot(dv)

			# Find the ideal input
			#l.ideal_x = dv.dot(l.Z.T)

			# Optimization
			eps_stable = 1e-8

			if beta_vv == 0:
				l.vv = dz
			else:
				l.vv  = beta_vv * l.vv + (1 - beta_vv) * dz
			
			if beta_sqr != 0:
				l.sqr = beta_sqr * l.sqr + (1 - beta_sqr) * np.square(dz)
				dz_optimized = l.lr * l.vv / (np.sqrt(l.sqr) + eps_stable)
			else:
				dz_optimized = dz

			# Modification based on input
			l.Z = l.Z + dz_optimized
			l.Z = l.Z / np.sqrt(np.sum(np.square(l.Z), axis=0))


	def __init__(grumpy_layer l, double lr = 0.006, double influence = 0.0005, double stable = 1e-10, double refresh = 1e-6):
		l.lr = lr
		l.influence = influence
		l.stable = stable
		l.refresh = refresh
		l.avg = 0.0

	cpdef set_scheme(grumpy_layer l, int input_size, np.ndarray choice_scheme, np.ndarray output_scheme):
		l.num_variations = choice_scheme.shape[0]
		l.num_choices = choice_scheme.max()+1
		l.num_outputs = output_scheme.max()+1

		l.Z = np.random.rand(input_size, l.num_variations).astype(np.float64)
		#l.Z = np.ones((input_size, l.num_variations))

		# Divide each prototype by the square root of its energy
		l.Z = l.Z / np.sqrt(np.sum(np.square(l.Z), axis=0))

		# Create a sqr matrix for adam optimization
		l.sqr = np.zeros_like(l.Z, dtype=np.float64)
		l.vv = np.zeros_like(l.Z, dtype=np.float64)

		l.input_size = input_size
		l.choice_scheme = choice_scheme
		l.output_scheme = output_scheme

	cpdef astype(grumpy_layer l, dtype):
		l.Z = l.Z.astype(dtype)
		l.sqr = l.sqr.astype(dtype)
		l.vv = l.vv.astype(dtype)

	cpdef update_lr(grumpy_layer l):
		l.lr *= (1 - l.avg)*l.influence + (1-l.influence)
		l.lr += l.stable

	cpdef refresh_lr(grumpy_layer l):
		l.lr += l.refresh
	
	cpdef set_refresh(grumpy_layer l, double refresh):
		l.refresh = refresh

	cpdef set_stable(grumpy_layer l, double stable):
		l.stable = stable

	cpdef set_influence(grumpy_layer l, double influence):
		l.influence = influence

	cpdef set_lr(grumpy_layer l, double lr):
		l.lr = lr

	cpdef double get_lr(grumpy_layer l):
		return l.lr

	cpdef int get_input_size(grumpy_layer l):
		return l.input_size

	cpdef int get_num_variations(grumpy_layer l):
		return l.num_variations

	cpdef int get_num_choices(grumpy_layer l):
		return l.num_choices

	cpdef int get_num_outputs(grumpy_layer l):
		return l.num_outputs

	cpdef double get_avg(grumpy_layer l):
		return l.avg

	cpdef np.ndarray get_o(grumpy_layer l):
		return l._o

	cpdef np.ndarray get_oi(grumpy_layer l):
		return l._oi

	cpdef get_Z(grumpy_layer l):
		return l.Z

	cpdef prune(grumpy_layer l, np.ndarray variation_indices):
		l.Z = np.delete(l.Z, variation_indices, axis=1)
		l.choice_scheme = np.delete(l.choice_scheme, variation_indices, axis=0)

"""
Notes ==========================================================================

Epochs vs Batches vs Outputs vs Choices vs Variations --------------------------

	If we were to compare the MNIST dataset to highschool exams, the number of
	epochs would be equivalent to the number of years you get held back and have
	to retake the same class.

	The number of batches would be how many exams you have for a class
	throughout the year (3 midterms and a final = 4 batches).

	Each element in a batch would correspond to one part of the exam (the exam
	is split up into parts A, B, and C taken on different days = 3 elements per
	batch [a.k.a batch size of 3]).

	The number of outputs is how many questions on each part (Part A, B, and C
	are each 50 questions long = 50 outputs).

	The number of choices is how many answers there are for each question
	(multiple choice A-D = 4 choices).

	The number of variations is how many prototypes you have in your head for a
	certain answer/choice (an apple could be green, yellow, or red = 3
	variations)

Footer
******

	Often you can use an output size of 1 and say each question is an element in
	a batch (batch size of 150 in our example). The distinction between outputs
	and batch elements is most evident in the hidden layer, where numerous
	outputs need to be calculated before they can be handed to the next layer to
	be interpereted.

	A variation size of 1 is often used for the hidden layer
"""
