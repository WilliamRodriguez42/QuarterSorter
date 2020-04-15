import pickle
import pygame
import lib.coin_center as cc
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

training_x = np.load('training_set/training_x.npy')
training_y = np.load('training_set/training_y.npy')
training_v = np.load('training_set/training_v.npy')

pygame.init()

display = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()

def change_contrast(image, factor):
	image = image - image.min()
	image = image / image.max()
	image = image.mean() + factor * (image - image.mean())
	image = np.clip(image, 0, 1)
	return image

def convert_pic(image):
	image = change_contrast(image, 1)
	#image = sobel(image)
	image = cc.imresize2(image, 700, 700)
	image *= 255
	image = image.astype(np.uint8)
	image = image.T[:, :, None]
	image = image.repeat(3, axis=2)
	return image

def update_surfs(i):
	global surf, text

	character_image = convert_pic(training_x[i])
	surf = pygame.surfarray.make_surface(character_image)
	text = font.render(str(i) + ' ' + str(chr(training_v[i] + ord('a'))), True, (255, 255, 255), (0, 0, 0))

def validate_i():
	global i
	while i < 0:
		i += training_x.shape[0]
	while i >= training_x.shape[0]:
		i -= training_x.shape[0]

i = 0
surf = None
text = None
font = pygame.font.Font(None, 32)
update_surfs(i)

stop = False

while not stop:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			stop = True

		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_RIGHT:
				i += 1
				validate_i()
				update_surfs(i)
			elif event.key == pygame.K_LEFT:
				i -= 1
				validate_i()
				update_surfs(i)

	display.fill((0, 0, 0))
	display.blit(surf, (50, 50))
	display.blit(text, (0, 0))

	pygame.display.update()
	clock.tick(60)

pygame.quit()
quit()
