import server.webapp_comm as wac

state_info = {}

state_info['New York'] = 12
state_info['Florida'] = 3
state_info['Nebraska'] = 22

wac.write_state_info(state_info)

print(wac.read_state_info())