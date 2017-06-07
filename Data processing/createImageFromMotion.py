"""
This script will analyze video files frame-by-frame. If motion between frames is detected,
the frame will be cropped and exported. If multiple motion is detected only the largest part
will be exported.
The results of this script can be used for transfer learning of a CNN.

Parts of this script are based on the "Basic motion detection and tracking with Python and OpenCV"
by Adrian Rosebrock. Available at:
http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
"""

import cv2
import time
import os
import sys
from threading import Thread
import argparse

localPath = "./"
saveFile = "saveFile.txt"
videoLength = 0

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="path to the video folder")
args = vars(ap.parse_args())

if args.get("path", None) is None:
	print("working local")
	relevant_path = localPath
else:
	print("working remote")
	relevant_path = args["path"]

included_extensions = ['AVI', 'mp4', 'm4v']
filenames = [files for files in os.listdir(relevant_path)
			if any(files.endswith(ext) for ext in included_extensions)]

'''
	Check if some video files in the directory were already processed. This allows user to
	pause this script.
'''
if(os.path.exists(saveFile)):
	filesDone = open(saveFile, "a+")
else:
	filesDone = open(saveFile, "w+")

with open(saveFile) as f:
	lines = f.readlines()

for title in filenames:
	for titlesDone in lines:
		titlesDone = titlesDone.strip("\n")
		if(title == titlesDone):
			print("File already done. Skipping "+title)
			print(filenames.index(title))
			filenames.pop(filenames.index(title))

print("Starting batch job. Grab a huuuuuuuuuge cup of coffee")

#Analyze every video file in the actual directory
for file in filenames:
	cap = cv2.VideoCapture(file)
	width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
	height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	totalFramesInFile = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frameCount = 0

	'''
		created the needed dirs for video conversion and remove processed video files
	'''
	videoOutputPath = relevant_path+"/images/"
	imageOutputPath = videoOutputPath+file+"/"
	if not os.path.exists(videoOutputPath):
		os.makedirs(videoOutputPath)
	if not os.path.exists(imageOutputPath):
		os.makedirs(imageOutputPath)
	outfile = videoOutputPath+'output_'+file+'.mp4'
	if(os.path.exists(outfile)):
		os.remove(outfile)

	out = cv2.VideoWriter(outfile, -1, cap.get(cv2.CAP_PROP_FPS), (width,height), True)

	'''
		print debug information about the file which is being processed.
	'''
	print(file)
	print(width, height)
	print(str(totalFramesInFile)+" @ "+str(fps))

	'''
		start a timer to give some metrics to the user
	'''
	start = time.time()
	end = time.time()
	firstFrame = None

	def imageWriter(path, frame, x, y, w, h):
		"""
		Exports a desired section of an image to the file system

		Keyword arguments:
		path -- target path for the export
		frame -- the frame to be cropped
		x -- x-axis coordinate
		y -- y-axis coordinate
		w -- width of the image section
		h -- height of the image section
		"""
		if(os.path.exists(path)):
				os.remove(path)
		'''
			its img[y: y + h, x: x + w], y-axis first
		'''
		frame = frame[y:y+h, x:x+w]
		cv2.imwrite(path, frame)

	'''
		analyze every video frame as long as there a frames left
	'''
	while(cap.isOpened()):
		ret, frame = cap.read()

		'''
			"warmup" --> make sure no invalid frame is being processed
		'''
		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (53, 53), 0)
			frameCount += 1
	 	'''
		 	if the first frame is None, initialize it
		'''
		if firstFrame is None:
			firstFrame = gray
			continue

		''' compute the absolute difference between the current frame and
		 	first frame
		'''
		frameDelta = cv2.absdiff(firstFrame, gray)
		thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
	 
		'''
			dilate the thresholded image to fill in possible holes and make shapes easier to detect,
			then find contours on this image
		'''
		thresh = cv2.dilate(thresh, None, iterations=40)
		(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
	 
		i = 0
		coordinations = {}
		'''
			process every contour found
		'''
		for c in cnts:
			'''
				if the contour is too small, ignore it
				"153" seems to be a good compromise for the given data
			'''
			if cv2.contourArea(c) < 153:
				continue
	 		'''
				compute the bounding box for the actual contour and draw it on the frame
			'''
			(x, y, w, h) = cv2.boundingRect(c)

			coordinations[i] = (x, y, w, h)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			i += 1
		if ret:
			'''
				make sure that contour coordinations are available for further processing
			'''
			if(coordinations != {}):
				tempResult = 0
				tempIndex = 0
				'''
					only use the biggest box in a frame
				'''
				for coord in coordinations:
					if(coordinations[coord][2] + coordinations[coord][3] >= tempResult):
						tempResult = coordinations[coord][2] + coordinations[coord][3]
						tempIndex = coord
				coordsToCut = coordinations[tempIndex]

				'''
					reopen the video file which is being processed to export a clean frame
					without the green markings
				'''
				cleanCap = cv2.VideoCapture(file)
				cleanCap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))
				cleanRet, cleanFrame = cleanCap.read()

				'''
					export the frame using a multithreaded approach for better performance
				'''
				if cleanRet:
					path = imageOutputPath+file+"_"+str(frameCount)+"_"+str(i)+".jpg"
					t = Thread(target=imageWriter, args=(path, cleanFrame, coordsToCut[0], coordsToCut[1], coordsToCut[2], coordsToCut[3]))
					t.start()
					t.join()
				cleanCap.release()
		else:
			break

		'''
			show the actual progress in the console to assure the script is still running
		'''
		end = time.time()
		videoLength = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
		print("Processing video "+str(filenames.index(file)+1)+" of "+str(len(filenames))+" - "+file+" - "+"{:0.0f}".format(frameCount/(totalFramesInFile)*100)+"%")

	'''
		Print the time it took to process the actual video and write its name in the save file. This
		ensures that, in case of a crash oder a user influenced break, this file won't be processed
		again.
	'''
	print("Processed: "+str(videoLength)+"s Video in "+str(end - start)+"s!")
	filesDone.write(file+"\n")

	'''
		Remove even the last processed video file and release everything when the job is done
	'''
	if(os.path.exists(outfile)):
				os.remove(outfile)
	cap.release()
	out.release()
filesDone.close()