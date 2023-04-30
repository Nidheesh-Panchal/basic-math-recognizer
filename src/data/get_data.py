import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
import cv2

base_dir = os.getcwd()

data_dir = os.path.join(base_dir, "src", "data")
# print(data_dir)

# download_url = "https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/download?datasetVersionNumber=9"
download_url = "https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols/download?datasetVersionNumber=4"

# download data to data_dir from the download url
download_file = "archive.zip"
if(download_file not in list(os.listdir(data_dir))):
	print("Please download dataset using the following link:")
	print(download_url)

data_zip_file = os.path.join(data_dir, download_file)
data_extract = os.path.join(data_dir, "extract")

if("extract" not in os.listdir(data_dir)):
	shutil.unpack_archive(data_zip_file, data_extract)

# now create .csv file from the extracted data

if("data.csv" in os.listdir(data_dir)):
    print("Data csv file already exists")
    exit()

data_dir = os.path.join(data_extract, "dataset")
# print(os.listdir(data_dir))

# set input image size to 28x28 with 1 channel (will take grayscale images)
batch_size = 128
img_row = 28
img_col = 28
channel = 1

# keep one fixed encoding for each math digit and operator, will use it for one hot encoding of the label
labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'add': 10, 'dec': 11, 'div': 12, 'eq': 13, 'mul': 14, 'sub': 15, 'x':16, 'y': 17, 'z': 18, '[': 19, ']': 20}
label = list(labels.keys())[:-2]

num_classes = len(label)
# print("Labels dict: ", labels)
# print("Labels list: ", label)
# print("Num of classes: ", num_classes)

# read image, take inverse, use threshold binary, merge all contours, make it close to square
inc_thresh = 0.6
def get_image(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = ~img
    _, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[0])

    a = int(28)
    b = int(28)
    x_max = np.Inf
    y_max = np.Inf
    w_max = 0
    h_max = 0
    
    l,w = 0,0
    
    for c in contour:
        x,y,a,b=cv2.boundingRect(c)
        
        x_max=min(x_max, x)
        y_max=min(y_max, y)
        w_max=max(x_max + w_max, x + a) - x_max
        h_max=max(y_max + h_max, y + b) - y_max

    add_x = 0
    add_y = 0
    if(w_max > h_max and h_max < inc_thresh * w_max):
        add_y = round((inc_thresh * w_max - h_max) * inc_thresh)
    if(h_max > w_max and w_max < inc_thresh * h_max):
        add_x = round((inc_thresh * h_max - w_max) * inc_thresh)
    
    x = max(0, x_max - 5 - add_x)
    y = max(0, y_max - 5 - add_y)
    xa = min(len(img[0]), x_max + w_max + 5 + add_x)
    yb = min(len(img), y_max + h_max + 5 + add_y)

    im_crop = thresh[y:yb, x:xa]
    im_resize = cv2.resize(im_crop,(28,28))
#     cv2.rectangle(img, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)
#     plt.imshow(img)
    im_resize = np.reshape(im_resize,(784))
    return im_resize

#create data
print("Progress: ")
dat = []
for folder in os.listdir(data_dir):
    if(folder == ".directory"):
        continue
    print("Label: ", folder)
    cat = labels[folder]
    for file in os.listdir(os.path.join(data_dir, folder)):
        if(file == ".directory"):
            continue

        row = get_image(os.path.join(data_dir, folder, file))
        row = np.append(row, cat)
        dat.append(row)

# create pandas dataframe
df = pd.DataFrame(dat)

df.to_csv(os.path.join(base_dir, "src", "data","data.csv"))