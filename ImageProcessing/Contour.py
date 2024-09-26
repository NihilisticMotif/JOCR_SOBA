import cv2
import numpy as np
from ImageUtility import OddKernelArea

def GetContours(dilate_img):
    contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def DetectContourImg(
        img:np.ndarray,
        threshold_px:None|int = None,
        kernel      :np.ndarray = np.ones((2,30)),
        kernel_area :int = 9,
        ):
    kernel_area = OddKernelArea(kernel_area)
    img = cv2.GaussianBlur(img,(kernel_area,kernel_area), 0) 
    if threshold_px != None:
        img = cv2.threshold(img, threshold_px, 255, cv2.THRESH_BINARY)[1]
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = cv2.dilate(img, kernel, iterations = 1)
    return img

def SortContours_Left2Right(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[0], reverse = is_reverse)

def SortContours_Top2Bottom(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[1], reverse = is_reverse)

def SortContours_Width(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[2], reverse = is_reverse)

def SortContours_Height(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[3], reverse = is_reverse)

def SortContours_Area(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.contourArea(x), reverse = is_reverse)

