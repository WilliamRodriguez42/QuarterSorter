from routes import *

if __name__ == '__main__':
	try:
		#app.run(host='0.0.0.0', port=5000)
		app.run()
	except Exception as e:
		print(e)
