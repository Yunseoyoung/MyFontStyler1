import os
import sys
from flask import Flask
app = Flask(__name__)
from flask import render_template, redirect

# 플라스크 서버를 구축하는 함수
# 웹기반 파이썬 프레임워크 : 플라스크
# 파이썬 내부에서 구현한 기능들을 웹에서 볼 수 있게끔 만들어주는 기능을 지원해줘요.
def server_run():
	# print(q.qsize())
	try:
		port = int(os.environ.get("PORT", 5000))
		app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False, threaded=True)
	except Exception as e:
		sys.stdout.write(str(e) + '\n')

@app.route('/')
def home():
	return render_template('main.html', result='nothing')

if __name__ == '__main__':
	server_run()
