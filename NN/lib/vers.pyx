import numpy as np
cimport numpy as np
from mnist import MNIST
from math import ceil, floor
from PIL import Image
import sys

import matplotlib.pyplot as plt

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

cpdef show_Z(
	grumpy_layer l,
	shape,
	output_index
	):
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
				vc.append(variations[:, j].reshape(shape))

			vo.append(vc)

	# Create a matrix to hold all the variations
	sub_im_width = shape[1]+10
	sub_im_height = shape[0]+10
	image = np.zeros((len(vo)*sub_im_height, max_variations_for_choice*sub_im_height))
	for i in range(len(vo)):
		variations = vo[i]
		for j in range(len(variations)):
			si = i*sub_im_height + 5
			ei = si + shape[0]
			sj = j*sub_im_width + 5
			ej = sj + shape[1]
			image[si:ei, sj:ej] = variations[j]

	imsave(image, 'Z.png')

cpdef feed(np.ndarray x, grumpy_layer l):
	cdef int batch_size = x.shape[0]
	cdef np.ndarray batch_iterator = np.arange(batch_size)
	cdef int i

	# Get value of each variation
	cdef np.ndarray _v = x.dot(l.Z) / x.shape[1] # (batch_size, num_variations)

	# Get best index and value of best variation for each choice
	cdef np.ndarray _c = np.zeros((batch_size, l.num_choices), dtype=np.float32) # (batch_size, num_choices)
	cdef np.ndarray _ci = np.zeros((batch_size, l.num_choices), dtype=np.uint32) # (batch_size, num_choices)
	cdef np.ndarray variation_mask

	for i in range(l.num_choices):
		variation_mask = (l.choice_scheme == i)
		_ci[:, i] = np.argmax(_v * variation_mask, axis=1)
		_c[:, i] = _v[batch_iterator, _ci[:, i]]

	# Get best index of best choice for each output and one hot best variation for best choice for each output
	cdef np.ndarray _oi = np.zeros((batch_size, l.num_outputs), dtype=np.uint32) # (batch_size, num_outputs)
	cdef np.ndarray _one_vc = np.zeros((batch_size, l.num_variations), dtype=np.uint32) # (batch_size, num_variations)
	cdef np.ndarray choice_mask
	for i in range(l.num_outputs):
		choice_mask = (l.output_scheme == i)
		_oi[:, i] = np.argmax(_c * choice_mask, axis=1)
		_one_vc[batch_iterator, _ci[batch_iterator, _oi[:, i]]] = 1

	l._v = _v
	l._c = _c
	l._ci = _ci
	l._oi = _oi
	l._one_vc = _one_vc

cpdef back(np.ndarray x, np.ndarray one_c, grumpy_layer l, beta=0.95, int update=True):
	# Need to get ci (index of best variation of best choice for each output) given one_c (one hot best choice for each output)
	cdef int batch_size = x.shape[0]
	cdef int num_variations = l.choice_scheme.shape[0]
	cdef int num_choices = l.choice_scheme.max()+1
	cdef int num_outputs = l.output_scheme.max()+1
	cdef np.ndarray batch_iterator = np.arange(batch_size)
	cdef int i, j

	cdef np.ndarray one_vc = np.zeros((batch_size, num_variations), dtype=np.uint32)
	cdef np.ndarray correct = np.zeros((batch_size, num_outputs), dtype=np.float32)
	cdef np.ndarray variation_mask = np.zeros((batch_size, num_variations), dtype=np.float32)
	cdef np.ndarray choice_mask, choices_for_output, choice_index, variations_for_best_choice, best_variation_index_for_choice

	for i in range(num_outputs):
		# Get best choice for this output
		choice_mask = (l.output_scheme == i)
		choices_for_output = one_c * choice_mask
		choice_index = np.argmax(choices_for_output, axis=1)
		correct[:, i] = np.equal(choice_index, l._oi[:, i])

		for j in range(batch_size):
			variation_mask[j] = (l.choice_scheme == choice_index[j])

		variations_for_best_choice = (l._v * variation_mask).T * one_c[batch_iterator, choice_index]
		best_variation_index_for_choice = np.argmax(variations_for_best_choice.T, axis=1)
		one_vc[batch_iterator, best_variation_index_for_choice] = 1

	# Find the average
	l.avg = np.mean(correct, axis=0)

	cdef np.ndarray dz_optimized
	if update:
		# Find the changes we need to make
		dv = one_vc.astype(np.int32) - l._one_vc.astype(np.int32)
		dz = x.T.dot(dv)

		# Find the ideal input
		l.ideal_x = dv.dot(l.Z.T)

		# Optimization
		eps_stable = 1e-15
		l.sqr = beta * l.sqr + (1 - beta) * np.square(dz)
		dz_optimized = l.lr * dz / (np.sqrt(l.sqr) + eps_stable)

		# Modification based on input
		l.Z = l.Z + dz_optimized
		l.Z = l.Z / np.sqrt(np.sum(np.square(l.Z), axis=0))

cdef class grumpy_layer:
	cdef float lr
	cdef float influence
	cdef float stable
	cdef float refresh
	cdef np.ndarray Z
	cdef np.ndarray sqr
	cdef np.ndarray choice_scheme
	cdef np.ndarray output_scheme
	cdef float avg
	cdef int num_variations
	cdef int num_choices
	cdef int num_outputs
	cdef int input_size
	cdef np.ndarray _v
	cdef np.ndarray _c
	cdef np.ndarray _ci
	cdef np.ndarray _oi
	cdef np.ndarray _one_vc
	cdef np.ndarray ideal_x

	def __init__(grumpy_layer self, float lr = 0.006, float influence = 0.0005, float stable = 1e-10, float refresh = 1e-6):
		self.lr = lr
		self.influence = influence
		self.stable = stable
		self.refresh = refresh
		self.avg = 0.0

	cpdef set_scheme(grumpy_layer self, int input_size, np.ndarray choice_scheme, np.ndarray output_scheme):
		self.num_variations = choice_scheme.shape[0]
		self.num_choices = choice_scheme.max()+1
		self.num_outputs = output_scheme.max()+1

		self.Z = np.random.rand(input_size, self.num_variations)
		#self.Z = np.ones((input_size, self.num_variations))

		# Divide each prototype by the square root of its energy
		self.Z = self.Z / np.sqrt(np.sum(np.square(self.Z), axis=0))

		# Create a sqr matrix for adam-like optimization
		self.sqr = np.zeros_like(self.Z)

		self.input_size = input_size
		self.choice_scheme = choice_scheme
		self.output_scheme = output_scheme

	cpdef update_lr(grumpy_layer self):
		self.lr *= (1 - self.avg)*self.influence + (1-self.influence)
		self.lr += self.stable

	cpdef refresh_lr(grumpy_layer self):
		self.lr += self.refresh

	cpdef int get_input_size(grumpy_layer self):
		return self.input_size

	cpdef int get_num_variations(grumpy_layer self):
		return self.num_variations

	cpdef int get_num_choices(grumpy_layer self):
		return self.num_choices

	cpdef int get_num_outputs(grumpy_layer self):
		return self.num_outputs

	cpdef float get_avg(grumpy_layer self):
		return self.avg

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
