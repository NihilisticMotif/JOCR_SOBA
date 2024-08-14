from PIL import Image 
from Utility import Numpy2Image, Image2Numpy
import numpy as np 
import cv2

def GrayImage(img):
    #img=Image2Numpy(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def InvertedImage(img):
    return cv2.bitwise_not(img)

def Rotate(img,angle):
    img = Numpy2Image(img)
    return img.rotate(angle)

def Sharpen(img):
    k2=-0.1
    k1=-5
    k0=-(k2*16+k1*8)+1
    kernel = np.array([
        [k2,k2,k2,k2,k2,], 
        [k2,k1,k1,k1,k2,], 
        [k2,k1,k0,k1,k2,], 
        [k2,k1,k1,k1,k2,], 
        [k2,k2,k2,k2,k2,]
        ])
    # https://youtu.be/KuXjwB4LzSA?si=mt-leKGKjpMnJGfg
    # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    return cv2.filter2D(img, -1, kernel)



