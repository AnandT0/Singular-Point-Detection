import os
import os.path
path1 = '/media/biometric/Data21/Core_Point/gTruth'    #path of folder of textfile
List_txt =  os.listdir(path1)
fw = open('gt_all_data_points.txt' , 'w+')
class1 = '0'
for file1 in List_txt:
	fo = open(path1 + '/' + file1 , "rw+")
	p = (file1.split('.')[0])[:-3] + ".bmp"
	print(p)
	line = fo.readlines()
	x,y = (line[0].strip()).split(" ")
	x = int(x)
	y  = int(y)
	x1 = y - 20
	y1 = x - 20	
	x2 = y + 20
	y2 = x + 20	
	str1 = p + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + class1 + "\n"
	fw.write(str1)
	
	fo.close()
	
fw.close()
