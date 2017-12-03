import numpy as np
import cv2
import os
import sys
from subprocess import Popen
from os import listdir
from os.path import isfile, join
import csv

def printMessage(msg):
	sys.stderr.write(msg)

darknet_dir = '/Users/Junwei/Desktop/Research/JackRabbot/darknet'
data_dir = '/Users/Junwei/Desktop/Research/JackRabbot/data/2DMOT2015/train'
frame_bbs_dir = '/Users/Junwei/Desktop/Research/JackRabbot/local_sort/bb_frames'
dets_dir = '/Users/Junwei/Desktop/Research/JackRabbot/local_sort/dets'

''' Process videos to save all frames. '''
# cap = cv2.VideoCapture('/Users/Junwei/Desktop/Research/JackRabbot/local_sort/TUD-Campus.mp4')
# index = 0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#     	cv2.imwrite(join(frame_dir, str(index) + ".jpg"), frame)
#     	index += 1
#     else:
#     	break
# cap.release()
# cv2.destroyAllWindows()

video_list = ['KITTI-13']

''' Invoke yolo to process all frames to get bounding boxes. '''
os.chdir(darknet_dir)
printMessage("------------------------------------- \n")
printMessage("Processing all videos.....\n")
printMessage("------------------------------------- \n")

for video in video_list:
	video_dir = join(data_dir, video)
	frame_dir = join(video_dir, 'img1')
	frames = [join(frame_dir, f) for f in listdir(frame_dir) if isfile(join(frame_dir, f))]
	print frames

	printMessage("------------------------------------- \n")
	index = 1
	for frame in frames:
		image_name = os.path.splitext(os.path.basename(frame))[0]
		print image_name
		printMessage("\nProcess frame " + str(index) + "......\n")
		frame_bb_dir = join(frame_bbs_dir, video)
		output_file = open(join(frame_bb_dir, 'frame_' + image_name + '_bb.txt'), 'w')
		child = Popen(['./darknet', 'detector',  'test', 'cfg/coco.data', 'cfg/yolo.cfg', 'yolo.weights', '-thresh', '0.3', frame], stdout=output_file)
		child.wait()
		output_file.close()
		index += 1

	''' Process all bounding boxes and feed it to sort. '''
	frame_bb_dir = join(frame_bbs_dir, video)
	frame_bbs = [join(frame_bb_dir, f) for f in listdir(frame_bb_dir) if isfile(join(frame_bb_dir, f)) and not f.startswith('.')]
	print frame_bbs
	
	tracking_dir = join(dets_dir, video)
	tracking_text_file = join(tracking_dir, 'det.txt')
	
	with open(tracking_text_file, 'wb') as tracking_file:
		wr = csv.writer(tracking_file, quoting=csv.QUOTE_NONE)
		index = 1
		for frame_bb in frame_bbs:
			with open(frame_bb, 'rb') as curr_frame:
				for line in curr_frame:
					row = [index, -1]
					bbox = [float(num) for num in line.split()]
					row.extend(bbox)
					row.extend([-1, -1, -1])
					print row
					wr.writerow(row)
			index += 1

''' Display in sort. '''
