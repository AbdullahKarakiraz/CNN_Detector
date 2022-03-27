from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
from Sliding_Window import sliding_window
from Image_Pyramid import image_pyramid

import numpy as np
import argparse
import imutils
import time
import cv2

parser = argparse.ArgumentParser()

parser.add_argument("-i","--image",required=True,help = "input image path")
parser.add_argument("-s","--size",type=str,default="(200,150)",help = "ROI size")
parser.add_argument("-c","--min-conf",type=float,default=0.9,help = "filtering threshold")
parser.add_argument("-v","--visualize",type=int,default=-1,help = "whether or not to show extra visualizations for debugging")

args = vars(parser.parse_args())

WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)


print("Loading Network...")
model = ResNet50(weights = "imagenet",include_top = True)

orig_img = cv2.imread(args["image"])
orig_img = imutils.resize(orig_img,width=WIDTH)
(H,W) = orig_img.shape[:2]

pyramid = image_pyramid(orig_img,scale=PYR_SCALE,minSize=ROI_SIZE)

rois = []
locs = []

start_t = time.time()

for image in pyramid:
    
    scale = W / float(image.shape[1])
    
    for (x,y,roiOrig) in sliding_window(image,step=WIN_STEP,ws=ROI_SIZE):
        
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        
        rois.append(roi)
        locs.append([x,y,x+w,y+h])
        
        if args["visualize"] > 0:
            
            clone = orig_img.copy()
            
            cv2.rectangle(clone,(x,y),(x+w,y+h),[0,255,0],2)
            cv2.imshow("Visualize",clone)
            cv2.imshow("ROI",roiOrig)
            cv2.waitKey(0)
            
stop_t = time.time()

print("looping over pyramid/windows took {:.5f} seconds".format(stop_t-start_t))

rois = np.array(rois, dtype="float32")        
            
print("Classifying ROIs...")
start_t = time.time()
preds = model.predict(rois)
stop_t = time.time()
print("Classifying ROIs took {:.5f} seconds".format(stop_t - start_t))

preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}


for (i,p) in enumerate(preds):
    
    (imagenetID, label, prob) = p[0]
    if prob >= args["min_conf"]:
        
        box = locs[i]
        
        L = labels.get(label,[])
        L.append((box,prob))
        labels[label] = L
        
for label in labels.keys():
    
    print("showing results for '{}'".format(label))
    clone = orig_img.copy()
    
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone,(startX,startY),(endX,endY),[0,255,0],2)
    
    cv2.imshow("Before", clone)
    clone = orig_img.copy()
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)
    
    for (startX, startY, endX, endY) in boxes:
        
        cv2.rectangle(clone, (startX, startY), (endX, endY),[0, 255, 0], 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, [0, 255, 0], 2)
        
    cv2.imshow("After", clone)
    cv2.waitKey(0)
    
    



























