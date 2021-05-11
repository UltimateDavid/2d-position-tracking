# USAGE
# python main.py

# import the necessary packages
from webcamvideostream import WebcamVideoStream
from fps import FPS
import imutils
import cv2 as cv
import math
import numpy as np
import time


font = cv.FONT_HERSHEY_SIMPLEX

min_area = 30 # minimum area of the ball (to avoid noise)
video_file = 'videos/example_03.mp4'

pause = False 

# Range of green color in HSV
lower = np.array([50,40,40])
upper = np.array([100,255,255])


# [ START VIDEOSTREAM ]
print("[INFO] sampling frames from webcam...")
# Check if we have a video or a webcam
if video_file is not None:
	stream = cv.VideoCapture(video_file)
# otherwise, we are reading from a video file
else:
	stream = WebcamVideoStream(src=0).start()
# measurements
fps = FPS().start()

listCenterX=[]
listCenterY=[]
listPoints=[]

draw = True

frame_width = 960 # int(stream.get(3))
frame_height = 544 # int(stream.get(4))

out = cv.VideoWriter('outframe.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
out2 = cv.VideoWriter('outmask.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height),isColor=False)

# [ START CAPTURING ]
# loop over every frame
while(True):
	key = cv.waitKey(10) & 0xFF
	if key== ord("q") or key == 27: break # quitting when ESCAPE or q is pressed
	if key== ord("h"): draw =not draw
	if key== ord(" "): pause =not pause # pausing while spacebar is pressed
	if(pause): continue


	# grab the frame from the stream and resize it
	(grabbed, frame_raw) = stream.read()
	if not(grabbed):
		break
	frame = imutils.resize(frame_raw, width=960, height=544)
	frame_raw = imutils.resize(frame_raw, width=960, height=544)

	# [ FIND GREEN PIXELS ]
	# Convert BGR to HSV
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    	# Threshold the HSV image to get only green colors
	mask = cv.inRange(hsv, lower, upper)


	# [ FIND OBJECTS ]
	# loop over the contours to find all objects
	contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)

	# loop over every object (we will probably only need one, but thats okay)

	# Find biggest contour (and in the mean time give every object a box)
	num = 0
	biggreen = None
	max_area = 0
	for c in contours:
		area = cv.contourArea(c)
		if area==0:
			continue
		if area > max_area:
			max_area = area
			biggreen = c
			pos = num
		num += 1
		
		# give a contour to every object
		if area > args["min_area"]:
			# calculate bounding box
			(xc, yc, wc, hc) = cv2.boundingRect(c)
			# draw
			cv2.rectangle(frame, (xc, yc), (xc + wc, yc + hc), (0, 150, 0), 1)

	if (	num > 0 # Check if there is any green object 
		and int(cv.contourArea(biggreen)) > min_area):	# no noise
		
		# [ CALCULATE POSITION ]
		# calculate bounding box
		(x, y, w, h) = cv.boundingRect(biggreen)

		# calculate x and y of observation
		xo = x + int(w/2)
		yo = y + int(h/2)
		
		listCenterX.append(xo) 
		listCenterY.append(yo)
		listPoints.append((xo,yo)) #((xo,yo,m))

		# DRAW
		if draw:
			cv.rectangle(frame, (x,  y), (x + w, y + h), (0, 255, 0), 4)
			cv.circle(frame, center=(xo, yo), radius=int((w+h)/24), color=(0, 0, 255), thickness=-1, lineType=8, shift=0)
			text = "Coordinates:" + str(xo) + ", " + str(yo)
			cv.putText(frame, format(text), (20, 30), font, 1, (0, 0, 255), 4)
		
		'''
		# Draw every measurement so far
		for n in range(len(listPoints)): 
			cv.circle(frame,(int(listPoints[n]),3, (0, 255, 0),-1)
		'''
	
	# [ DISPLAY ]
	# check to see if the frame should be displayed to our screen
	cv.imshow('Frameraw',frame_raw)
	cv.imshow('Frame',frame)
	cv.imshow('Vision',mask)
	out.write(frame)
	out2.write(mask)
	# update the FPS counter
	fps.update()

	
# [ CLEARING UP ]
# stop the timer and display information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#print("Measurements: "+ str(listPoints))

stream.release()
out.release()
out2.release()
cv.destroyAllWindows()
