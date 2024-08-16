import numpy as np 
import cv2

def ThinFont(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img

def ThickFont(img):
    kernel = np.ones((2,2),np.uint8)
    img = cv2.bitwise_not(img)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img