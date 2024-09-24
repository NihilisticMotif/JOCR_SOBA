import numpy as np
import cv2

def RemoveNoise(img):
    # https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb
    # https://www.geeksforgeeks.org/erosion-and-dilation-morphological-transformations-in-opencv-in-cpp/
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    return (img)

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

def Dilate(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)
    
def Erode(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(img, kernel, iterations = 1)

def Opening(img,kernel=None):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def Canny(img):
    return cv2.Canny(img, 100, 200)

# https://nanonets.com/blog/ocr-with-tesseract/