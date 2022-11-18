from multiprocessing import freeze_support
import os
from flask import render_template, redirect, request, url_for
from run import server_run, app
from werkzeug.utils import secure_filename
from glob import glob
from preprocess import link_src_trg
from train import run_train
from train import make_bmp
from train import bmp_convert_and_makefont as run_total

global save_bmp
save_bmp = 'static/results/handwiting_fonts_myfont'
@app.route('/', methods=['GET', 'POST'])
def main():
	return render_template('main.html')


@app.route('/fileUpload', methods=['GET', 'POST'])
def fileUpload():
        try:
                f = request.files['file']
                os.makedirs('static/uploads/', exist_ok=True)
                f.save('static/uploads/' + secure_filename(f.filename))
                files = os.listdir("static/uploads")

                # 파일명과 파일경로를 데이터베이스에 저장함
                response = "INSERT INTO images (image_name, image_dir) VALUES ('%s', '%s')" % (
                secure_filename(f.filename), 'uploads/' + secure_filename(f.filename))
        except Exception as e:
                print(e)
        return view()

@app.route('/view', methods=['GET', 'POST'])
def view():
	data = glob(os.path.join("static/uploads", '*'))
	data_list = []

	for obj in data:  # 튜플 안의 데이터를 하나씩 조회해서
		data_dic = {  # 딕셔너리 형태로
			# 요소들을 하나씩 넣음
			'name': f'uploads/{os.path.basename(obj)}'
		}
		print(data_dic)
		data_list.append(data_dic)  # 완성된 딕셔너리를 list에 넣음

	return render_template('view.html', data_list=data_list)  # html을 렌더하며 DB에서 받아온 값들을 넘김

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
	link_src_trg()

	return render_template('view2.html',
						   data_list=[f'handwritings/realimg_with_srcfont/{x}' for x in list(filter(lambda x:x if x[-3:]=='png' else None, os.listdir('static/handwritings/realimg_with_srcfont')))])

@app.route('/train', methods=['GET', 'POST'])
def train():
        try:
                epoch = request.form['epoch']

                print(epoch)
                epoch = int(epoch)
        except:
                epoch = 10000
        run_train(epoch)

        return render_template('view2.html',
                data_list=[f'fixed_real_fake/{x}' for x in list(filter(lambda x:x if x[-3:]=='png' else None, os.listdir('static/fixed_real_fake')))][-10:])

@app.route('/remove', methods=['GET', 'POST'])
def remove():
        data_list = glob(os.path.join("static/uploads", "*"))
        [os.remove(file) for file in data_list]

        return render_template('main.html')


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    global save_bmp
    font_name = request.form['font_name']
    save_bmp = make_bmp(font_name)
    print(save_bmp)
    return render_template('main.html')

@app.route('/make_font', methods=['GET', 'POST'])
def make_font():
    global save_bmp
    print(f"save_bmp : {save_bmp}")
    run_total(save_bmp)
    return render_template('main.html')

from flask import send_file
@app.route('/download')
def download():
    global save_bmp
    bmp_path = os.path.basename(save_bmp)
    ttf_name = bmp_path.split('_')[-1]
    file_path =f'static/results/fonts/{ttf_name}.ttf'
    return send_file(file_path, attachment_filename=f'{ttf_name}.ttf', as_attachment=True)


if __name__ == "__main__":
	freeze_support()
	# 플라스크 서버를 실행시킨다.
	server_run()
