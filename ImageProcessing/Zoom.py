import cv2 
import numpy as np
from ImageUtility import GetSize
# https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb

def RemoveBorders(img:np.ndarray):
    contours, heiarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]
    return (crop)

def Zoom(img:np.ndarray, zoom:int = 1):
    # https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
    # zoom < 1 implies Zoom out
    # zoom > 1 implies Zoom in
    rot_mat = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 0, abs(zoom))
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def CreateBorders(img:np.ndarray, size:int = 50):
    top, bottom, left, right = [abs(size)]*4
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT) # , value=color)
    return img

def Crop(img:np.ndarray, x:int|None = None, y:int|None = None, width:int|None = None, height:int|None = None):
    x = GetSize(x, img.shape[1])
    y = GetSize(y, img.shape[0])
    if type(width) == int:
        width = GetSize(x + width, img.shape[1], img.shape[1] - x)
    else:
        width = img.shape[1]
    if type(height) == int:
        height = GetSize(y + height, img.shape[0], img.shape[0] - y)
    else:
        height = img.shape[0]
    return img[y:y+height, x:x+width]
