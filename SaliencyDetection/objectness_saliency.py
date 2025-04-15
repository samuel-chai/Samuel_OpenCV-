# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-n", "--max-detections", type=int, default=10, help="maximum # of detections to examine")
args = vars(ap.parse_args())
# load the input image
image = cv2.imread(args["image"])

# initialize OpenCV's objectness saliency detector and set the path
# to the input model files
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# compute the bounding box predictions used to indicate saliency
(success, saliencyMap) = saliency.computeSaliency(image)

saliencyMap = (saliencyMap * 255).astype("uint8")
threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow("Saliency Map", saliencyMap)
cv2.imshow("Threshold Map", threshMap)
cv2.waitKey(0)
