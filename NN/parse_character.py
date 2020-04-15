import pygame
import lib.coin_center as cc
import numpy as np
import bisect
import glob
import pickle
from id_stor.manage_id_stor import write_id_stor, read_id_stor, create_id_stor

pygame.init()

def convert_pic(image, width, height):
	image = cc.imresize2(image, height, width)
	image = image.T[:, :, None]
	image = image.repeat(3, axis=2)
	image *= 255
	image = image.astype(np.uint8)
	return image

def convert_unwrapped(image):
	image = convert_pic(image, image_width, image_height)
	return image

def get_pics(i):
	global image_height, image_hw_ratio, image_scale
	top_unwrapped, bot_unwrapped = cc.parse_dataset_element(i)

	if image_height is None:
		image_hw_ratio = top_unwrapped.shape[0] / top_unwrapped.shape[1]
		image_height = int(image_width * image_hw_ratio)
		image_scale = top_unwrapped.shape[1] / image_width

	return top_unwrapped, bot_unwrapped

def update_surf():
	global text, top_unwrapped, bot_unwrapped, image_index, surf_select, surf
	top_unwrapped, bot_unwrapped = get_pics(image_index)

	if surf_select == 0:
		surf = pygame.surfarray.make_surface(convert_unwrapped(top_unwrapped))
	else:
		surf = pygame.surfarray.make_surface(convert_unwrapped(bot_unwrapped))
	text = font.render(str(image_index), True, (255, 255, 255), (0, 0, 0))

def get_cropped_image():
	rect_width_image_scale = int(rect_width * image_scale + 0.5)
	rect_height_image_scale = int(rect_height * image_scale + 0.5)
	sx_image_scale = int(x * image_scale + 0.5)
	sy_image_scale = int(y * image_scale + 0.5)
	ex_image_scale = sx_image_scale + rect_width_image_scale
	ey_image_scale = sy_image_scale + rect_height_image_scale

	unwrapped = top_unwrapped
	if surf_select == 1:
		unwrapped = bot_unwrapped

	cropped = unwrapped[sy_image_scale:ey_image_scale, sx_image_scale:ex_image_scale]
	return cropped

def update_reference_surf():
	global reference_surf
	index = ord(c) - ord('a')
	reference_surf = pygame.surfarray.make_surface(convert_pic(reference_images[index], 500, 500))
	reference_surf.set_alpha(125)

def set_reference_surf():
	global reference_images
	# Update reference images
	index = ord(current_id[1]) - ord('a')
	reference_images[index] = get_cropped_image()
	update_reference_surf()

def update_zoomed_surf():
	global zoomed_surf
	cropped = get_cropped_image()
	zoomed_surf = pygame.surfarray.make_surface(convert_pic(cropped, 500, 500))

def draw_rect(x, y, w, h, color=(255, 0, 0)):
	pygame.draw.line(display, color, (x, y), (x+w, y))
	pygame.draw.line(display, color, (x+w, y), (x+w, y+h))
	pygame.draw.line(display, color, (x+w, y+h), (x, y+h))
	pygame.draw.line(display, color, (x, y+h), (x, y))

def draw_id(xy, c, wh, color=(255, 0, 0)):
	global image_y

	x, y = xy
	rect_width, rect_height = wh
	px = int(x + 0.5)
	py = int(y + 0.5)

	draw_rect(px, py + image_y, rect_width, rect_height, color=color)
	tc_text = tc_font.render(str(c), True, color, (0, 0, 0))
	display.blit(tc_text, (px, py-10 + image_y))

def set_retro_i():
	global nx, ny, image_width, image_height, c, retro_just_set, retro_i
	while retro_i >= len(ids):
		retro_i -= len(ids)
	while retro_i < 0:
		retro_i += len(ids)

	(rx, ry), rc, (rrect_width, rrect_height) = ids[retro_i]

	nx = rx
	ny = ry
	c = rc
	rect_width = rrect_width
	rect_height = rrect_height

	retro_just_set = True

def update_current_id():
	global current_id, previous_id, x, y, c, rect_width, rect_height, reference_images
	previous_id = current_id
	current_id = ((x, y), c, (rect_width, rect_height))
	index = ord(c) - ord('a')
	if reference_images is not None and reference_images[index] is None:
		set_reference_surf()

def delete_retro_i(new_retro_i=-1):
	global ids, retro_i
	ids.pop(retro_i)
	retro_i = new_retro_i

def insort_current_id():
	global current_id, ids, on_top
	if not on_top:
		bisect.insort(ids, current_id)

def save_to_id_stor():
	global id_stor, ids, surf_select, image_index
	id_stor[image_index][surf_select] = ids

def load_from_id_stor():
	global id_stor, ids, surf_select, image_index
	ids = id_stor[image_index][surf_select]

image_index = 0
surf_select = 0
surf = None
zoomed_surf = None
reference_surf = None
top_unwrapped = None
bot_unwrapped = None
text = None
font = pygame.font.Font(None, 32)
tc_font = pygame.font.Font(None, 24)

image_y = 520
image_width = 1600
image_height = None
image_hw_ratio = None
image_scale = None

display = pygame.display.set_mode((image_width, 600))
clock = pygame.time.Clock()
update_surf()

stop = False

ids = []
current_id = None
previous_id = None
retro_i = -1
p_retro_i = -1
pp_retro_i = 0
on_top = False

retro_just_set = False
rect_width = 30
rect_height = 30

x = 0
y = 0
nx = 0
ny = 0
c = 'a'
speed = 1 / image_scale
reference_images = None
update_zoomed_surf()
update_current_id()
update_current_id()

if len(glob.glob('id_stor/reference_images.pkl')) != 0:
	reference_images = pickle.load(open('id_stor/reference_images.pkl', 'rb'))
else:
	reference_images = np.empty(26, dtype=object)
	for i in range(len(reference_images)):
		reference_images[i] = None

id_stor = create_id_stor(cc.dataset.shape[0])
if len(glob.glob('id_stor/id_stor.txt')) != 0:
	read_id_stor(id_stor)
	load_from_id_stor()

while not stop:
	retro_just_set = False

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			stop = True
			save_to_id_stor()
			write_id_stor(id_stor)
			pickle.dump(reference_images, open('id_stor/reference_images.pkl', 'wb+'))

		if event.type == pygame.MOUSEBUTTONUP:
			nx, ny = pygame.mouse.get_pos()
			ny = ny - image_y - rect_width // 2
			nx = nx - rect_width // 2

		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LSHIFT:
				speed = rect_width / 4
			elif event.key == pygame.K_LEFT:
				nx = x - speed
			elif event.key == pygame.K_RIGHT:
				nx = x + speed
			elif event.key == pygame.K_UP:
				ny = y - speed
			elif event.key == pygame.K_DOWN:
				ny = y + speed
			elif event.key == pygame.K_TAB:
				set_reference_surf()
			elif event.key == pygame.K_RETURN:
				if retro_i == -1:
					insort_current_id()
					retro_i = -1
				else:
					retro_i = -1

			elif event.key == 46: # >
				if retro_i == -1:
					insort_current_id()
					retro_i = pp_retro_i
				retro_i += 1
				set_retro_i()

			elif event.key == 44: # <
				if retro_i == -1:
					insort_current_id()
					retro_i = pp_retro_i
				retro_i -= 1
				set_retro_i()

			elif event.key == 8: # Delete
				if retro_i != -1:
					delete_retro_i(retro_i-1)
					if len(ids) != 0:
						retro_i = len(ids)-1
						set_retro_i()
				else:
					if pp_retro_i < len(ids):
						retro_i = pp_retro_i
						set_retro_i()

			elif event.key == 49 or event.key == 50: # One
				save_to_id_stor()
				surf_select = 1 - surf_select
				retro_i = -1
				update_surf()
				update_zoomed_surf()
				load_from_id_stor()
			elif event.key == 61 or event.key == 45: # + -
				save_to_id_stor()
				surf_select = 0
				retro_i = -1
				if event.key == 61:
					image_index += 1
				else:
					image_index -= 1
				while image_index < 0:
					image_index += cc.dataset.shape[0]
				while image_index >= cc.dataset.shape[0]:
					image_index -= cc.dataset.shape[1]

				update_surf()
				update_zoomed_surf()
				load_from_id_stor()

			else:
				s = pygame.key.name(event.key)
				if len(s) == 1:
					o = ord(s[0])
					if o >= 0x61 and o <= 0x7a:
						c = s[0]

		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_LSHIFT:
				speed = 1 / image_scale

	keys = pygame.key.get_pressed()
	if keys[pygame.K_LEFT]:
		nx = x - speed
	if keys[pygame.K_RIGHT]:
		nx = x + speed
	if keys[pygame.K_UP]:
		ny = y - speed
	if keys[pygame.K_DOWN]:
		ny = y + speed

	if nx < 0:
		nx = 0
	elif nx > image_width - rect_width:
		nx = image_width - rect_width
	if ny < 0:
		ny = 0
	elif ny > image_height - rect_height:
		ny = image_height - rect_height

	if nx != x or ny != y:
		x = nx
		y = ny

		update_zoomed_surf()

	update_current_id()
	if previous_id != current_id and retro_i != -1 and not retro_just_set:
		delete_retro_i()

	px = int(x+0.5)
	py = int(y+0.5)

	display.fill((0, 0, 0))
	display.blit(surf, (0, image_y))
	display.blit(text, (0, 0))

	on_top = False
	for i in range(len(ids)):
		id = ids[i]
		color = (255, 255, 0)
		if int(id[0][0] + 0.5) == px and int(id[0][1] + 0.5) == py:
			color = (255, 102, 0)
			on_top = True
		if retro_i == i:
			color = (0, 0, 255)
		draw_id(*id, color=color)

	if not on_top and retro_i == -1:
		draw_id((x, y), c, (rect_width, rect_height), color=(255, 0, 0))

	if retro_i != p_retro_i:
		pp_retro_i = p_retro_i
		p_retro_i = retro_i

		if pp_retro_i == -1:
			pp_retro_i = 0

	display.blit(zoomed_surf, (image_width//2 - 250 - 300, 10))
	index = ord(c) - ord('a')
	if reference_images[index] is not None:
		update_reference_surf()
		display.blit(zoomed_surf, (image_width//2 - 250 + 300, 10))
		display.blit(reference_surf, (image_width//2 - 250 + 300, 10))

	pygame.display.update()
	clock.tick(60)

pygame.quit()
quit()
