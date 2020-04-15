def write_id_stor(id_stor, location='id_stor/id_stor.txt'):
	s = ""

	for i in range(len(id_stor)):
		for j in range(2):
			if id_stor[i][j] is not None:
				for id in id_stor[i][j]:
					for arg in id:
						if len(arg) > 1:
							for a in arg:
								s += str(a) + ','
						else:
							s += str(arg) + ','
					s += '\n'
			s += 'END OF IMAGE IDS\n'
		s += 'END OF COIN IDS\n'

	output_file = open(location, 'w+')
	output_file.write(s)
	output_file.close()

def read_id_stor(id_stor, location='id_stor/id_stor.txt'):
	input_file = open(location, 'r')
	s = input_file.read()
	input_file.close()

	coin_sections = s.split('END OF COIN IDS\n')
	for i, coin_section in enumerate(coin_sections):
		image_sections = coin_section.split('END OF IMAGE IDS\n')
		for j, image_section in enumerate(image_sections):
			lines = image_section.split('\n')
			for line in lines:
				args = line.split(',')
				if len(args) == 1:
					continue

				x = float(args[0])
				y = float(args[1])
				c = args[2]
				w = float(args[3])
				h = float(args[4])
				id = ((x, y), c, (w, h))
				id_stor[i][j].append(id)

def create_id_stor(length):
	id_stor = []
	for i in range(length):
		id_stor.append([[], []])
	return id_stor
