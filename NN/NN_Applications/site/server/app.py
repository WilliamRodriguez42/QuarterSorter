import numpy as np
import time
from PIL import Image
import glob

def get_frame(camera_name):
	while True:
		file_name = '../../dataset/' + camera_name

		try:
			Image.open(file_name)
			im_file = open(file_name, 'rb')
			frame = im_file.read()
			im_file.close()


			yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
					frame +
					b'\r\n')

			time.sleep(1/2) # 2 fps
		except:
			pass