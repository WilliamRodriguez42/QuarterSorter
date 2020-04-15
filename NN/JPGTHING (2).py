from serial import Serial, SerialException
import sys
import time
import glob

existing_files = glob.glob('dataset/*')
max_num = 0
for file_name in existing_files:
	end_of_num = file_name.find('.')
	num = int(file_name[8:end_of_num])
	if max_num < num:
		max_num = num
current_i = max_num + 1

msp_ser = Serial("/dev/ttyAMA0", 2000000)

use_ard = False

if use_ard:
	ard_ser = Serial("/dev/tty.usbmodem14201", 9600)
	time.sleep(5)
	ard_ser.write(b'p')

while True:

	try:
		result = b''
		num_read = 0
		while True:
			to_read = msp_ser.inWaiting()
			num_read += to_read

			if num_read % 500 == 0:
				sys.stdout.write("Read {} bytes\r".format(num_read))
				sys.stdout.flush()

			if to_read:
				phrase = msp_ser.read(to_read)
				result += phrase

				if b'eatmybooty' in result and b'Done reading FIFO.\r\n' in result:
					print()

					start = -1
					if b'eatmybooty1:' in result:
						start = result.find(b'eatmybooty1:') + len(b'eatmybooty1:')
						file_name = 'dataset/{}.b.jpg'.format(current_i)

					if b'eatmybooty2:' in result:
						start2 = result.find(b'eatmybooty2:') + len(b'eatmybooty2:')

						if start2 < start or start == -1:
							start = start2
							file_name = 'dataset/{}.t.jpg'.format(current_i)

					if current_i % 2 == 0 and use_ard:
						ard_ser.write(b'a')
						time.sleep(2)
						ard_ser.write(b'p')

					remaining = result[start:]
					end = remaining.find(b'...')

					print(result[start:start+end])

					res_start = start+end+len(b'...')
					res_end = result.find(b'Done reading FIFO.\r\n')
					jpeg_bytes = result[res_start:res_end]

					output = open(file_name, 'wb+')
					output.write(jpeg_bytes)
					output.close()

					result = result[res_end + len(b'Done reading FIFO.\r\n'):]
					num_read = 0

					current_i += 1

	except Exception as e:
		print(e)
		break
