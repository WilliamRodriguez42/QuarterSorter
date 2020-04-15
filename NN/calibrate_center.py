
import pygame
import lib.coin_center as cc
import numpy as np
import pickle

pygame.init()

display = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

def convert_pic(image):
	image = cc.imresize(image, 0.4)
	image = image.T[:, :, None]
	image = image.repeat(3, axis=2)
	image *= 255
	image = image.astype(np.uint8)
	return image

def update_surfs(i):
	global surf, x, y
	im = convert_pic(cc.show_image_center(1 - cc.dataset[0, i], y, x))
	surf = pygame.surfarray.make_surface(im)

x = 0
y = 0
i = 0
surf = None
update_surfs(i)

stop = False
mult = 1

while not stop:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			stop = True

		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_b:
				i = 1 - i
				update_surfs(i)
			if event.key == pygame.K_RETURN:
				center = np.array([y, x], dtype=np.float32)

				if i == 0:
					pickle.dump(center, open('pickle/top_center.pkl', 'wb+'))
				else:
					pickle.dump(center, open('pickle/bot_center.pkl', 'wb+'))
				print('saved')
			if event.key == pygame.K_LSHIFT:
				mult = 20
		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_LSHIFT:
				mult = 1

	keys = pygame.key.get_pressed()
	if keys[pygame.K_LEFT]:
		x -= 1 * mult
		print(y, x)

	if keys[pygame.K_RIGHT]:
		x += 1 * mult
		print(y, x)

	if keys[pygame.K_UP]:
		y -= 1 * mult
		print(y, x)

	if keys[pygame.K_DOWN]:
		y += 1 * mult
		print(y, x)

	update_surfs(i)

	display.fill((0, 0, 0))
	display.blit(surf, (100, 100))

	pygame.display.update()
	clock.tick(60)

pygame.quit()
quit()
