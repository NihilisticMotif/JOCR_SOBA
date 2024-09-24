import cv2

def GrayImage(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def InvertedImage(img):
    return cv2.bitwise_not(img)
