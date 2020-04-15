
import pygame
import lib.coin_center as cc
import numpy as np
import pickle
from PIL import Image
import io

pygame.init()

display = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

def convert_pic(image):
	image = cc.imresize(image, 0.4)
	image = image.T[:, :, None] / image.max()
	image = image.repeat(3, axis=2)
	image *= 255
	image = image.astype(np.uint8)
	return image

def update_surfs(i):
	global surf, x, y
	# if i == 0:
	# 	im = convert_pic(cc.show_image_center(top_image, y, x))
	# else:
	# 	im = convert_pic(cc.show_image_center(bot_image, y, x))
	im = convert_pic(cc.show_image_center(cc.dataset[0, i] * 255, y, x))
	surf = pygame.surfarray.make_surface(im)

file = open('dataset/1.t.jpg', 'rb')
content = file.read()
file.close()

image = Image.open(io.BytesIO(content)).convert('L')
top_image = np.array(image, dtype=np.float32)

file = open('dataset/2.b.jpg', 'rb')
content = file.read()
file.close()

image = Image.open(io.BytesIO(content)).convert('L')
bot_image = np.array(image, dtype=np.float32)

center = pickle.load(open('pickle/top_center.pkl', 'rb'))
y = center[0]
x = center[1]
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

				if i == 0:
					center = pickle.load(open('pickle/top_center.pkl', 'rb'))
				else:
					center = pickle.load(open('pickle/bot_center.pkl', 'rb'))

				y = center[0]
				x = center[1]
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
	clock.tick(20)

pygame.quit()
quit()
