import os
from model import predict
from flask import Flask, app, jsonify, render_template, request, url_for, redirect, flash

base_dir = os.getcwd()
src_dir = os.path.join(base_dir, "src")
model_dir = os.path.join(src_dir, "model")
data_dir = os.path.join(src_dir, "data")
test_dir = os.path.join(src_dir, "test")

# predict.predict(os.path.join(test_dir, "eq7.png"))

UPLOAD_FOLDER = '/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods = ["GET"])
def home():
	return render_template("home.html")

@app.route("/predict", methods = ["POST"])
def req_predict():
	print(request.files.keys())
	return render_template("home.html")

if __name__=="__main__":
	app.run(debug=True)