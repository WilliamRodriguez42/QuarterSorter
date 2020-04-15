import pygame
import lib.coin_center as cc
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter, sobel

pygame.init()

display = pygame.display.set_mode((800, 600))
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
	image = cc.imresize(image, 0.2)
	image *= 255
	image = image.astype(np.uint8)
	image = image.T[:, :, None]
	image = image.repeat(3, axis=2)
	return image

def get_pics(i):
	top_unwrapped, bot_unwrapped = cc.parse_dataset_element(i)

	top_unwrapped = convert_pic(top_unwrapped)
	bot_unwrapped = convert_pic(bot_unwrapped)

	return top_unwrapped, bot_unwrapped

def update_surfs(i):
	global top_surf, bot_surf, text
	top_unwrapped, bot_unwrapped = get_pics(i)
	top_surf = pygame.surfarray.make_surface(top_unwrapped)
	bot_surf = pygame.surfarray.make_surface(bot_unwrapped)
	text = font.render(str(i), True, (255, 255, 255), (0, 0, 0))

i = 0
top_surf = None
bot_surf = None
text = None
font = pygame.font.Font(None, 32)
update_surfs(i)

surf_size = top_surf.get_size()
surf_mid = surf_size[1] / 2

top_y = 250
bot_y = 300

stop = False
mouse_pos = pygame.mouse.get_pos()
mouse_y = mouse_pos[1]

face_index = pickle.load(open('pickle/face_index.pkl', 'rb'))
if face_index.shape[0] != cc.dataset.shape[0]:
	face_index = np.zeros(cc.dataset.shape[0])

while not stop:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			stop = True

		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT:
				i -= 1
				if i < 0:
					i += cc.dataset.shape[0]

				update_surfs(i)

			elif event.key == pygame.K_RIGHT:
				i += 1
				if i >= cc.dataset.shape[0]:
					i -= cc.dataset.shape[0]

				update_surfs(i)

			elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
				face_index[i] = 1 - face_index[i]
				pickle.dump(face_index, open('pickle/face_index.pkl', 'wb+'))

	if face_index[i] == 0:
		y = top_y - 5
	else:
		y = bot_y - 5

	display.fill((0, 0, 0))
	red_square = pygame.draw.rect(display, (255, 0, 0), (0, y, surf_size[0]+10, surf_size[1]+10))
	display.blit(top_surf, (5, top_y))
	display.blit(bot_surf, (5, bot_y))
	display.blit(text, (0, 0))

	pygame.display.update()
	clock.tick(60)

pygame.quit()
quit()
