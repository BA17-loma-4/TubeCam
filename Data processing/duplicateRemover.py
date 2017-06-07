'''
As the script createImageFromMotion.py creates a lot of data, this script here is needed to clean
up. Using nearly the same logic as in the other file it will move all duplicates to a "deleted"
folder. It will not actually delete any file for safety reasons.
'''
import cv2
import os
import shutil
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="path to root folder of the images that should be cleaned up")
args = vars(ap.parse_args())

if args.get("path", None) is None:
	print("working local")
	relevant_path = "."
else:
	print("working remote")
	relevant_path = args["path"]

included_extensions = ['jpg']

'''
	Recursively step through every folder in given root path
'''
for root, dirs, files in os.walk(relevant_path, topdown=False):
	for name in dirs:
		'''
			Ignore already "deleted" images
		'''
		if("deleted" not in name):
			print("working in directory: "+name)
			name = relevant_path+"/"+name
			'''
				Only work with desired files
			'''
			filenames = [files for files in os.listdir(name)
				if any(files.endswith(ext) for ext in included_extensions)]
			oldPic = ""
			delDir = "./deleted"

			if not os.path.exists(delDir):
				os.makedirs(delDir)

			'''
				Read every image in directory using OpenCV
			'''
			for file in filenames:
				image = cv2.imread(name+"/"+file)

				height, width, channels = image.shape

				'''
					Remove images that are too small and would scale badly.
					Inception v3 uses 299x299x3
				'''
				if((height*width < 62500) or ((width < 200) or (height < 200))):
					shutil.move(name+"/"+file, delDir+"/"+file)
				else:
					'''
						Detect motion between the last and the actual file using the same logic
						as in createImageFromMotion.py
						If no motion is detected move the actual image to the "deleted" directory
					'''
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					gray = cv2.GaussianBlur(gray, (25, 25), 0)

					if(oldPic != ""):
						gray = cv2.resize(gray, (299, 299))
						oldPic = cv2.resize(oldPic, (299, 299))

						frameDelta = cv2.absdiff(oldPic, gray)
						thresh = cv2.threshold(frameDelta, 20, 25, cv2.THRESH_BINARY)[1]
						
						thresh = cv2.dilate(thresh, None, iterations=40)
						(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

						if(cnts == []):
							shutil.move(name+"/"+file, delDir+"/"+file)
					else:
						oldPic = gray