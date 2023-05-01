import os
from model import predict
from flask import Flask, app, jsonify, render_template, request, url_for, redirect, flash, send_from_directory
from distutils.log import debug
from fileinput import filename

base_dir = os.getcwd()
src_dir = os.path.join(base_dir, "src")
model_dir = os.path.join(src_dir, "model")
data_dir = os.path.join(src_dir, "data")
test_dir = os.path.join(src_dir, "test")
template_dir = os.path.join(src_dir, "templates")

# predict.predict(os.path.join(test_dir, "eq7.png"))

UPLOAD_FOLDER = os.path.join(template_dir, "uploads")
if("uploads" not in os.listdir(template_dir)):
	os.mkdir(os.path.join(template_dir, "uploads"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods = ["GET"])
def home():
	return render_template("home.html")

# @app.route("/predict", methods = ["POST"])
# def req_predict():
# 	print(request.files.keys())
# 	return render_template("home.html")
  
@app.route('/', methods = ['POST'])  
def req_predict():
	for file in os.listdir(app.config['UPLOAD_FOLDER']):
		os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
	f = request.files['file']
	print(f)
	# pred = predict.predict(f.filename)
	# print(pred)
	path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
	f.save(path)
	pred = predict.predict(path)
	print(path)
	new_file_path = "uploads/" + f.filename.split(".")[-2] + "_1." + f.filename.split(".")[-1]
	# os.remove(path)
	return render_template("home.html", path = new_file_path, pred = pred)

@app.route('/uploads/<filename>', methods=["GET"])
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
  
if __name__ == '__main__':  
	app.run(debug=True)