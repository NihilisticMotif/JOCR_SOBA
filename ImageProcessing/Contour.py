import cv2
from GrayImage import ColorImage, GrayImage 
import numpy as np 
from Convolution import GaussBlur
from Threshold import BinaryPx, OtsuBinaryPx
from ShowImage import SaveImage

#####################################################################################################################

def GetContours(dilate_img):
    contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    return contours

def GetLargestContour(dilate_img):
    return GetContours(dilate_img)[0]

#####################################################################################################################

def DefaultDilateImage(img,threshold_px=None,is_binary_inv=False,kernel = np.ones((2,30))):
    dilate_img = GrayImage(img)
    dilate_img = GaussBlur(dilate_img,9)
    if isinstance(threshold_px, (int,float)):
        dilate_img = BinaryPx(dilate_img,threshold_px,isinverse=is_binary_inv)   
    dilate_img = OtsuBinaryPx(dilate_img,isinverse=True)   
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate_img = cv2.dilate(dilate_img, kernel, iterations=2)
    return dilate_img

#####################################################################################################################

def DrawContours(img,contour,rgb=(255,0,0)):
    img = ColorImage(img)
    img = cv2.drawContours(img, contour, -1, rgb, 3)
    return img 

def DrawDilateContours(dilate_img,img=None,rgb=(255,0,0)):
    contours = GetContours(dilate_img)
    if not isinstance(img,np.ndarray):
        return DrawContours(img=dilate_img,contour=contours,rgb=rgb)
    return DrawContours(img=img,contour=contours,rgb=rgb)

def DrawDefaultContours(img,rgb=(255,0,0),threshold_px=None,is_binary_inv=False,kernel = np.ones((2,30)),is_original_img=True):
    dilate_img = DefaultDilateImage(
        img,
        threshold_px=threshold_px,
        is_binary_inv=is_binary_inv,
        kernel = kernel)
    if is_original_img==True:
        return DrawDilateContours(dilate_img=dilate_img,img=img,rgb=rgb)
    else:
        return DrawDilateContours(dilate_img=dilate_img,img=None,rgb=rgb)

#####################################################################################################################
