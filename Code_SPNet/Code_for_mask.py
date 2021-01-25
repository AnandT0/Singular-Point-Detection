import cv2
import numpy as np

file1 = open('gt_all_data.txt')
files = file1.readlines()
for fil in files:
	fil1 = fil.split(",")
	img = cv2.imread('/media/biometric/Data21/Core_Point/data_used/' + fil1[0])
	mask = np.zeros(img.shape[:2],dtype="uint8")
	cv2.rectangle(mask,(int(fil1[1]),int(fil1[2])),(int(fil1[3]),int(fil1[4])),255,-1)
	output = '/media/biometric/Data21/Core_Point/Mask_gt/' + (fil1[0])
	cv2.imwrite(output,mask)	

