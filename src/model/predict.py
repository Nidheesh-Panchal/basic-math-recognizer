import numpy as np
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('predict')

base_dir = os.getcwd()
src_dir = os.path.join(base_dir, "src")
model_dir = os.path.join(src_dir, "model")
data_dir = os.path.join(src_dir, "data")
test_dir = os.path.join(src_dir, "test")

batch_size = 128
img_row = 28
img_col = 28
channel = 1

labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'add': 10, 'dec': 11, 'div': 12, 'eq': 13, 'mul': 14, 'sub': 15, 'x':16, 'y': 17, 'z': 18}
label = list(labels.keys())

num_classes = len(label)

train_model = os.path.join(model_dir, "model.h5")
model = tf.keras.models.load_model(train_model)

# print("Model summary: ")
# print(model.summary())

eq_dir = test_dir
eq_list = os.listdir(eq_dir)
eq_list.sort()
# print(eq_list)

def contour_union(x, y, xa, yb, key, l):
	x_i = x[key]
	y_i = y[key]
	xa_i = xa[key]
	yb_i = yb[key]
	for j in l:
		x_i = min(x_i, x[j])
		y_i = min(y_i, y[j])
		xa_i = max(xa_i, xa[j])
		yb_i = max(yb_i, yb[j])
	return [x_i, y_i, xa_i, yb_i]

def merge_all_contours(img, contour):
	contours = []
	contours_x = []
	contours_xa = []
	contours_y = []
	contours_yb = []

	for c in contour:
		x,y,a,b=cv2.boundingRect(c)
		contours.append([x, y, a, b])
		contours_x.append(x)
		contours_xa.append(x + a)
		contours_y.append(y)
		contours_yb.append(y + b)
		# cv2.rectangle(img, (x - 5, y - 5), (x + a + 5, y + b + 5), (0, 255, 0), 2)
	
	overlaps={}

	for i in range(len(contours)):
		for j in range(i + 1, len(contours)):
			if((contours_x[i] <= contours_x[j] and contours_xa[i] >= contours_x[j] and contours_xa[i] <= contours_xa[j]) or # i on left of j
			   (contours_x[i] >= contours_x[j] and contours_x[i] <= contours_xa[j] and contours_xa[i] >= contours_xa[j]) or # j on left of i
			   (contours_x[i] >= contours_x[j] and contours_xa[i] <= contours_xa[j]) or # i inside of j
			   (contours_x[i] <= contours_x[j] and contours_xa[i] >= contours_xa[j])): # j inside of i
				if(i not in overlaps.keys()):
					overlaps[i] = [j]
				else:
					overlaps[i].append(j)

	# print(overlaps)

	for key in reversed(list(overlaps.keys())):
		for ival in overlaps[key]:
			keep = []
			if(ival not in overlaps.keys()):
				continue
			for jval in overlaps[ival]:
				if(jval not in overlaps[key]):
					overlaps[key].append(jval)
			overlaps[ival] = []
	# print(overlaps)
	
	keys = list(overlaps.keys())
	used_contours = []
	max_contour = []
	for key in keys:
		if(len(overlaps[key]) == 0):
			overlaps.pop(key)
			continue
		overlaps[key].sort()
		used_contours.append(key)
		used_contours.extend(overlaps[key])
		max_contour.append(contour_union(contours_x, contours_y, contours_xa, contours_yb, key, overlaps[key]))

	# print(overlaps)
	# print(max_contour)
	# print(used_contours)
		
	new_contour = []
	for i in range(len(contours)):
		if(i in used_contours):
			continue
		x_i = contours_x[i]
		y_i = contours_y[i]
		xa_i = contours_xa[i]
		yb_i = contours_yb[i]
		new_contour.append([x_i, y_i, xa_i, yb_i])

	# print(new_contour)

	new_contour.extend(max_contour)
	new_contour.sort(key = lambda x: x[0])
	
	return new_contour

def adjust_contours(img, new_contour, dims_thresh, inc_thresh):
	new_new_contour = []
	# print(img)
	for c in new_contour:
		x,y,xa,yb=c
		l, w = (xa-x), (yb-y)
		contour_dims = (xa-x) * (yb-y)
		# print("countour dimension: ", contour_dims)
		if(l < dims_thresh and w < dims_thresh):
			continue
		add_x = 0
		add_y = 0
		if(l > w and w < inc_thresh * l):
			add_y = round((inc_thresh * l - w) * inc_thresh)
		if(w > l and l < inc_thresh * w):
			add_x = round((inc_thresh * w - l) * inc_thresh)
		# print(add_x, add_y)
		if(contour_dims > dims_thresh):
			# adding contour
			# print("adding contour")
			
			x = max(0, x - 5 - add_x)
			y = max(0, y - 5 - add_y)
			xa = min(len(img[0]), xa + 5 + add_x)
			yb = min(len(img), yb + 5 + add_y)
			new_new_contour.append([x,y,xa,yb])
	return new_new_contour

def get_image_all(file):
	img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	logger.info(f"Size of the original image {len(img[0])} x {len(img)}")
	logger.info("Use grayscale image and flip the colors so that the image becomes white text and black background")
	img = ~img

	logger.info("Use threshold binary to get sharp edges")
	_, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

	logger.info("Get contours")
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	logger.info("Sort contours based on x-axis location")
	contour = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[0])
	
	logger.info("merge all the x-axis overlapping contours")
	new_contour = merge_all_contours(img, contour)
	
	logger.info(f"Merged contours : {new_contour}")
	
	dims_thresh = 20
	# remove small contours
	# make contours slightly square
	# increasing the width or length threshold
	inc_thresh = 0.6

	logger.info(f"Use threshold for per dimension: {dims_thresh} and width, height ratio threshold: {inc_thresh}")
	
	final_contour = adjust_contours(img, new_contour, dims_thresh, inc_thresh)

	logger.info(f"Final contours : {final_contour}")

	images = []

	logger.info("Get individual contour images for recognition and add contour rectangles on original image")
	for c in final_contour:
		x,y,xa,yb=c
		cv2.rectangle(img, (x, y), (xa, yb), (255, 255, 255), 2)
		images.append(thresh[y : yb, x : xa])
	images = np.array(images)
	# plt.imshow(img)

	return images, img

def process_image(img):
	logger.info(f"Shape of the image : {img.shape}")
	temp = cv2.resize(img, (img_row, img_col))
	temp = temp / 255
	temp = np.reshape(temp, (img_row, img_col, channel))
	# return temp
	return np.array([temp])

def predict(img):
	eq_img, cnt_img = get_image_all(img)
	new_path = img.split(".")[-2]+"_1." + img.split(".")[-1]
	logger.info(f"Saving individual character boxes image at new path: {new_path}")
	cv2.imwrite(new_path, ~cnt_img)
	eq_str = ""
	for imgs in eq_img:
		im = process_image(imgs)
		pred = model.predict(im, verbose=0)[0]
		lab = label[pred.argmax()]

		# print("Predicted label : ", lab)
		
		if(lab == "eq"):
			lab = "="
		elif(lab == "dec"):
			lab = "."
		elif(lab == "add"):
			lab = "+"
		elif(lab == "sub"):
			lab = "-"
		elif(lab == "div"):
			lab = "/"
		elif(lab == "mul"):
			lab = "*"

		eq_str += lab
	logger.info(f"Recognized Equation: {eq_str}")

	return eq_str