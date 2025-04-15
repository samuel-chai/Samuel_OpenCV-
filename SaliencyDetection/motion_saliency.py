# import the necessary packages
from imutils.video import VideoStream
import imutils
import time
import cv2
# initialize the motion saliency object and start the video stream
saliency = None
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# if our saliency object is None, we need to instantiate it
	if saliency is None:
		saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	# convert the input frame to grayscale and compute the saliency
	# map based on the motion model
	(success, saliencyMap) = saliency.computeSaliency(frame)
	saliencyMap = (saliencyMap * 255).astype("uint8")
	threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# display the image to our screen
	cv2.imshow("Frame", frame)
	cv2.imshow("Map", saliencyMap)
	cv2.imshow("Threshold", threshMap)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()