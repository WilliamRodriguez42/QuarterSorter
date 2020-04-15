from serial import Serial, SerialException
import sys
import time
import glob
import categorize_state as cs
import webapp_comm as wac

# existing_files = glob.glob('dataset/*')
# max_num = 0
# for file_name in existing_files:
# 	end_of_num = file_name.find('.')
# 	num = int(file_name[8:end_of_num])
# 	if max_num < num:
# 		max_num = num
# current_i = max_num + 1
current_i = 1

stm_ser = Serial("/dev/ttyAMA0", 2000000)

# use_ard = False

# if use_ard:
# 	ard_ser = Serial("/dev/ttyACM0", 9600)
# 	time.sleep(5)
# 	ard_ser.write(b'p')

result = b''
num_read = 0
prev_num_read = 0

prev_num_read = [0] * 10
start_time = time.time()

max_state_num = 0
max_state_response = 0
max_state_name = 0

state_info = {}

while True:
	time.sleep(0.1)

	to_read = stm_ser.inWaiting()
	num_read += to_read

	if prev_num_read[-1] == 0 and num_read != 0:
		start_time = time.time()

	if num_read == prev_num_read[0] and num_read != 0:
		result = b''
		num_read = 0
	prev_num_read.append(num_read)
	del prev_num_read[0]

	sys.stdout.write("Read {} bytes\r".format(num_read))
	sys.stdout.flush()

	if to_read:
		phrase = stm_ser.read(to_read)
		result += phrase

		if b'eatmybooty' in result and b'Done reading FIFO.\r\n' in result:
			print()

			if b'ACK Capture' in result:
				temp = result.find(b'ACK Capture')
			if b'Done reading FIFO' in result:
				temp = result.find(b'Done reading FIFO')

			start = -1
			top = True
			if b'eatmybooty1:' in result:
				start = result.find(b'eatmybooty1:') + len(b'eatmybooty1:')
				file_name = 'dataset/{}.b.jpg'.format(2)
				top = False

			if b'eatmybooty2:' in result:
				start2 = result.find(b'eatmybooty2:') + len(b'eatmybooty2:')

				if start2 < start or start == -1:
					start = start2
					file_name = 'dataset/{}.t.jpg'.format(1)

			remaining = result[start:]
			end = remaining.find(b'...')

			res_start = start+end+len(b'...')
			res_end = result.find(b'Done reading FIFO.\r\n')
			jpeg_bytes = result[res_start:res_end]

			output = open(file_name, 'wb+')
			output.write(jpeg_bytes)
			output.close()

			stop_time = time.time()
			print(stop_time - start_time)
			start_time = stop_time

			result = result[res_end + len(b'Done reading FIFO.\r\n'):]
			num_read = 0

			state_num, state_response, state_name = cs.categorize(jpeg_bytes, top=top)
			if state_response > max_state_response:
				max_state_response = state_response
				max_state_name = state_name
				max_state_num = state_num

			if current_i % 2 == 0:
				if max_state_response > 0.95:
					print(max_state_name)

					if max_state_name not in state_info:
						state_info[max_state_name] = 0
					state_info[max_state_name] += 1

					wac.write_state_info(state_info)

				max_state_response = 0

			# if current_i % 2 == 0 and use_ard:
			# 	ard_ser.write(b'a')
			# 	time.sleep(2)
			# 	ard_ser.write(b'p')

			current_i += 1
