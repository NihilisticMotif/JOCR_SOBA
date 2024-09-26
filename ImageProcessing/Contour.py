import cv2
import numpy as np
from ImageUtility import OddKernelArea, GetDefaultOption

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

def SortContours(contour:list, is_reverse:bool = False, method:str|int = 4):
    message = '''
method:str|int = 4
* 0, x, left, right
* 1, y, top, bottom, up, down
* 2, w, width
* 3, h, height
* 4, size, area, s, a
'''
    if isinstance(method, int):
        method = GetDefaultOption(method,[4,0,1,2,3],message)
        if method == 0:
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[0], reverse = is_reverse)
        if method == 1:
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[1], reverse = is_reverse)
        if method == 2:
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[2], reverse = is_reverse)
        if method == 3:
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[3], reverse = is_reverse)
        if method == 4:
            return sorted(contour, key=lambda x:cv2.contourArea(x), reverse = is_reverse)
    if isinstance(method, str):
        method = method.replace(" ", "").lower()
        if any(sub == method[0] for sub in ['0', 'x', 'l', 'r']):
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[0], reverse = is_reverse)
        if any(sub == method[0] for sub in ['1', 'y', 't', 'b', 'u', 'd']):
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[1], reverse = is_reverse)
        if any(sub == method[0] for sub in ['2', 'w']):
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[2], reverse = is_reverse)
        if any(sub == method[0] for sub in ['3', 'h']):
            return sorted(contour, key=lambda x:cv2.boundingRect(x)[3], reverse = is_reverse)
        if any(sub == method[0] for sub in ['4', 's', 'a']):
            return sorted(contour, key=lambda x:cv2.contourArea(x), reverse = is_reverse)
        else:
            print(message)
            return sorted(contour, key=lambda x:cv2.contourArea(x), reverse = is_reverse)
