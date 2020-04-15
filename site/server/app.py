import cv2
import numpy as np
import time

def get_frame(camera_name):
	while True:
		mat = np.random.rand(100, 100, 3) * 255
		_, frame = cv2.imencode('.jpg', mat)

		yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
				frame.tobytes() +
				b'\r\n')

		time.sleep(1/15) # 15 fps