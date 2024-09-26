import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing_View'
sys.path.insert(1, ImageProcessingPath)

import numpy as np
import cv2
from ImageUtility import SetPx
from Blur import GaussBlur
from Threshold import Threshold

def RemoveNoise(img:np.ndarray):
    # https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb
    # https://www.geeksforgeeks.org/erosion-and-dilation-morphological-transformations-in-opencv-in-cpp/
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    return (img)

def ThinFont(img:np.ndarray):
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img

def ThickFont(img:np.ndarray):
    kernel = np.ones((2,2),np.uint8)
    img = cv2.bitwise_not(img)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    return img

def Dilate(img:np.ndarray, kernel:np.ndarray = np.ones((5,5),np.uint8)):
    return cv2.dilate(img, kernel, iterations = 1)
    
def Erode(img:np.ndarray, kernel:np.ndarray = np.ones((5,5),np.uint8)):
    return cv2.erode(img, kernel, iterations = 1)

def Opening(img:np.ndarray, kernel:np.ndarray = np.ones((5,5),np.uint8)):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def Canny(img:np.ndarray, c1:int = 100, c2:int = 200):
    if c1 > c2:
        tp = c1
        c1 = c2 
        c2 = tp    
    return cv2.Canny(img, SetPx(c1), SetPx(c2))

# https://nanonets.com/blog/ocr-with-tesseract/