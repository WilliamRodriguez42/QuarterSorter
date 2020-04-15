from flask import Flask, Response, request, jsonify, abort
import os
import glob
from app import *
import webapp_comm as wac
import json
import re

app = Flask(__name__)

def json_dumps(form):
	content = json.dumps(form, sort_keys=True, indent=4)
	content = re.sub('\n +', lambda match: '\n' + '\t' * (len(match.group().strip('\n')) // 3), content)
	return content

@app.route('/<path:path>', methods=['GET'])
def send_whatever(path):
	ext = path[path.rfind('.')+1:]
	file = open('../client/' + path, 'rb')
	content = file.read()
	file.close()

	if ext == 'ico':
		response = Response(content, mimetype='image/vnd.microsoft.icon')

	elif ext == 'css':
		response = Response(content, mimetype="text/css")

	elif ext == 'svg':
		response = Response(content, mimetype="image/svg+xml")

	else:
		response = Response(content)

	return response

@app.route('/video_feed')
def video_feed():
	return Response(
		get_frame(request.args["camera_name"]),
		mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def send_home():
	html = open('../client/index.html', 'r')
	content = html.read()
	html.close()

	return Response(content)

@app.route('/state_info', methods=['GET'])
def state_info():
	return Response(json_dumps(wac.read_state_info()))

@app.route('/move_quarter_up_one_step', methods=['POST'])
def move_quarter_up_one_step():
	wac.step_up()
	return Response("Ok")

@app.route('/move_quarter_down_one_step', methods=['POST'])
def move_quarter_down_one_step():
	wac.step_down()
	return Response("Ok")