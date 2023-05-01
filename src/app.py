import os
import logging
from model import predict
from flask import Flask, app, jsonify, render_template, request, url_for, redirect, flash, send_from_directory
from distutils.log import debug
from fileinput import filename

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger('main')
logger.info('Starting app')


base_dir = os.getcwd()
src_dir = os.path.join(base_dir, "src")
model_dir = os.path.join(src_dir, "model")
data_dir = os.path.join(src_dir, "data")
test_dir = os.path.join(src_dir, "test")
template_dir = os.path.join(src_dir, "templates")

# predict.predict(os.path.join(test_dir, "eq7.png"))

logger.info("Check if uploads folder already exists")

UPLOAD_FOLDER = os.path.join(template_dir, "uploads")
if("uploads" not in os.listdir(template_dir)):
	logger.info("Uploads folder does not exist, creating folder")
	os.mkdir(os.path.join(template_dir, "uploads"))
	logger.info("Uploads folder created")

logger.info("Starting flask app")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods = ["GET"])
def home():
	logger.info("Home page")
	return render_template("home.html")

# @app.route("/predict", methods = ["POST"])
# def req_predict():
# 	print(request.files.keys())
# 	return render_template("home.html")
  
@app.route('/', methods = ['POST'])  
def req_predict():
	logger.info("Request received for prediciton")
	logger.info("Deleting any previous images stored in upload folder")
	for file in os.listdir(app.config['UPLOAD_FOLDER']):
		os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
	logger.info("Deletion of previous uploaded file complete")
	f = request.files['file']
	logger.info(f"Uploaded file : {f}")
	# pred = predict.predict(f.filename)
	# print(pred)
	logger.info("Saving uploaded image to uploads folder")
	path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
	f.save(path)
	logger.info("Image saved")

	logger.info("Sending image for prediction")
	pred = predict.predict(path)
	logger.info(f"Received prediction : {pred}")

	logger.info("Creating new file path for contour image")
	new_file_path = "uploads/" + f.filename.split(".")[-2] + "_1." + f.filename.split(".")[-1]
	logger.info(f"New path : {new_file_path}")

	return render_template("home.html", path = new_file_path, pred = pred)

@app.route('/uploads/<filename>', methods=["GET"])
def upload(filename):
    logger.info("Request to GET contour image")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
  
if __name__ == '__main__':  
	app.run(debug=True)