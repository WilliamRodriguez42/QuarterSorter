STATE_INFO_FILE_PATH = '/home/pi/Desktop/NN_Applications/site/resources/state_info.txt'
STEP_QUARTER_FILE_PATH = '/home/pi/Desktop/NN_Applications/site/resources/step_quarter.txt'

def write_state_info(state_info):
    ssi = sorted(state_info.items())

    content = ""
    for t in ssi:
        content += '{}\t{}\n'.format(*t)

    output_file = open(STATE_INFO_FILE_PATH, 'w+')
    output_file.write(content)
    output_file.close()

def read_state_info():
    state_info = {}

    input_file = open(STATE_INFO_FILE_PATH, 'r')
    content = input_file.read()
    input_file.close()

    lines = content.split('\n')

    for line in lines:
        if line == "": break

        parts = line.split('\t')

        state_name = parts[0]
        state_count = int(parts[1])

        state_info[state_name] = state_count
    
    return state_info

def step_up():
    f = open(STEP_QUARTER_FILE_PATH, 'r')
    content = f.read()
    f.close()

    content += 'step up\n'

    f = open(STEP_QUARTER_FILE_PATH, 'w')
    f.write(content)
    f.close()

def step_down():
    f = open(STEP_QUARTER_FILE_PATH, 'r')
    content = f.read()
    f.close()

    content += 'step down\n'

    f = open(STEP_QUARTER_FILE_PATH, 'w')
    f.write(content)
    f.close()

def read_step_count():
    f = open(STEP_QUARTER_FILE_PATH, 'r')
    content = f.read()
    f.close()

    step_count = 0

    lines = content.split('\n')
    for line in lines:
        if line == 'step up':
            step_count += 1
        elif line == 'step down':
            step_count -= 1

    return step_count