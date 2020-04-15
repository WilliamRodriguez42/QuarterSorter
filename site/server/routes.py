from flask import Flask, Response, request, jsonify, abort
import os
import glob
from app import *

app = Flask(__name__)

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
