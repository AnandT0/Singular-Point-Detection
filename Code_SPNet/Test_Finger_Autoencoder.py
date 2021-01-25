'''Import the libraries'''
import os
import cv2
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy
import numpy.random as rng
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#from skimage.transform import resize
#from skimage.io import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
'''Set Keras image format '''
K.set_image_data_format('channels_last')


###########################################  Encoder  ####################################################


def Encoder(input_img):

	Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1")(input_img)
	Econv1_1 = BatchNormalization()(Econv1_1)
	Econv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same',  name = "block1_conv2")(Econv1_1)
	Econv1_2 = BatchNormalization()(Econv1_2)
	pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_2)
	
	Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1")(pool1)
	Econv2_1 = BatchNormalization()(Econv2_1)
	Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2")(Econv2_1)
	Econv2_2 = BatchNormalization()(Econv2_2)
	pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

	Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1")(pool2)
	Econv3_1 = BatchNormalization()(Econv3_1)
	Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2")(Econv3_1)
	Econv3_2 = BatchNormalization()(Econv3_2)
	pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)

	encoded = Model(input = input_img, output = [pool3, Econv1_2, Econv2_2, Econv3_2] )

	return encoded
#########################################  Bottleneck ##################################################
#
##
def neck(input_layer):

	Nconv = Conv2D(256, (3,3),padding = "same", name = "neck1" )(input_layer)
	Nconv = BatchNormalization()(Nconv)
	Nconv = Conv2D(128, (3,3),padding = "same", name = "neck2" )(Nconv)
	Nconv = BatchNormalization()(Nconv)

	neck_model = Model(input_layer, Nconv)
	return neck_model
#
##########################################  Decoder   ##################################################

def Decoder(inp ):

	up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_1")(inp[0])
	up1 = BatchNormalization()(up1)
	up1 = merge([up1, inp[3]], mode='concat', concat_axis=3, name = "merge_1")
	Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1")(up1)
	Upconv1_1 = BatchNormalization()(Upconv1_1)
	Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2")(Upconv1_1)
	Upconv1_2 = BatchNormalization()(Upconv1_2)

	up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2")(Upconv1_2)
	up2 = BatchNormalization()(up2)
	up2 = merge([up2, inp[2]], mode='concat', concat_axis=3, name = "merge_2")
	Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1")(up2)
	Upconv2_1 = BatchNormalization()(Upconv2_1)
	Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2")(Upconv2_1)
	Upconv2_2 = BatchNormalization()(Upconv2_2)
	
	up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3")(Upconv2_2)
	up3 = BatchNormalization()(up3)
	up3 = merge([up3, inp[1]], mode='concat', concat_axis=3, name = "merge_3")
	Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
	Upconv3_1 = BatchNormalization()(Upconv3_1)
	Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
	Upconv3_2 = BatchNormalization()(Upconv3_2)
	   
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)
	convnet = Model(input = inp, output =  decoded)
	return convnet

#########################################################################################################

##########################################'''Initialise the model.'''####################################

x_shape = 256
y_shape = 256
channels = 1
input_img = Input(shape = (x_shape, y_shape,channels))

#Encoder
encoded = Encoder(input_img)	#return encoded representation with intermediate layer Pool3(encoded), Econv1_3, Econv2_3,Econv3_3

#Decoder
HG_ = Input(shape = (x_shape/(2**3),y_shape/(2**3),128))
conv1_l = Input(shape = (x_shape,y_shape,16))
conv2_l = Input(shape = (x_shape/(2**1),y_shape/(2**1),64))
conv3_l = Input(shape = (x_shape/(2**2),y_shape/(2**2),128))
decoded = Decoder( [HG_, conv1_l, conv2_l, conv3_l])

#BottleNeck
Neck_input = Input(shape = (x_shape/(2**3), y_shape/(2**3),128))
neck = neck(Neck_input)

#Combined
output_img = decoded([neck(encoded(input_img)[0]), encoded(input_img)[1], encoded(input_img)[2], encoded(input_img)[3]])
model= Model(input = input_img, output = output_img )
model.summary()
model.compile(optimizer = Adam(0.0005), loss='binary_crossentropy', metrics = ["accuracy"])
model.load_weights('/media/biometric/Data21/Core_Point/Model_exp/UNet/Stats/UNet.h5')

#########################################################################################################

name = os.listdir("Test_Original")
input_images = []
output_images = []
print("loading_images")
count = 0
for i in name : 
	print(i)
	img_x1 = cv2.imread("Test_Original/"+i,0)
	#print(img_x)	
	img_x = cv2.resize(img_x1, (256,256)) /255.0
	img_x = img_x.reshape(1,256,256,1)
	img_x = img_x.reshape(1,256,256,1)
	pred = model.predict(img_x)[0][:,:,0]
	pred = pred.reshape(256,256,1)
	path_out = "/media/biometric/Data21/Core_Point/Test_Result2/" + str(i)
	pred1=pred *255
	pred1 = np.uint8(pred1)
        edges = cv2.Canny(pred1,25,30,L2gradient=True)
	#sobelx64f = cv2.Sobel(pred1,cv2.CV_64F,1,0,ksize=5)
	#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	#laplacian = cv2.Laplacian(pred1,cv2.CV_64F)
	#abs_sobel64f = np.absolute(sobelx64f)
	#sobel_8u = np.uint8(abs_sobel64f)
	#dst = cv2.add(img_x1,edges)
	cv2.imwrite(path_out,pred1)
	
