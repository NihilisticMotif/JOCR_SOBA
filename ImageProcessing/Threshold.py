from PIL import Image 
from Utility import Numpy2Image
import numpy as np 
import cv2
from EditImage import GrayImage

'''
img = Input Image 
* (PIL.Image.Image or np.ndarray)

px = Pixel
'''
########################################################################################################################################################

def BinaryPx(img,threshold_px,max_px=255,IsInverse=False):
    img=GrayImage(img)
    if IsInverse==False:
        # if px > threshold_px then px = max_px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_BINARY)[1]
    else:
        # if px < threshold_px then px = max_px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_BINARY_INV)[1]

def AdaptiveBinaryPx01(img,area,C=2,max_px=255,IsInverse=False):
    # Calculating the mean value of the surround pixels within the square area (width=area) and use that mean value as the threshold_px
    if IsInverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,area,C)

def AdaptiveBinaryPx02(img,area,C=2,max_px=255,IsInverse=False):
    # Calculating the "Gaussian" value of the surround pixels within the square area (width=area) and use that mean value as the threshold_px
    if IsInverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,area,C)

########################################################################################################################################################

def TruncatedPx(img,threshold_px,max_px=255):
    img=GrayImage(img)
    # if px > threshold_px then px = threshold_px else px = px
    return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_TRUNC)[1]

def AdaptiveTruncatedPx01(img,area,C=2,max_px=255,IsInverse=False):
    return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_TRUNC,area,C)

def AdaptiveTruncatedPx02(img,area,C=2,max_px=255,IsInverse=False):
    return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_TRUNC,area,C)

########################################################################################################################################################

def Set0Px(img,threshold_px,max_px=255,IsInverse=False):
    img=GrayImage(img)
    if IsInverse==False:
        # if px > threshold_px then px = px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_TOZERO)[1]
    else:
        # if px < threshold_px then px = px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_TOZERO_INV)[1]

def AdaptiveSet0Px01(img,area,C=2,max_px=255,IsInverse=False):
    if IsInverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_TOZERO,area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_TOZERO_INV,area,C)

def AdaptiveSet0Px02(img,area,C=2,max_px=255,IsInverse=False):
    if IsInverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_TOZERO,area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_TOZERO_INV,area,C)

'''

1. Thresholding technique by GreeksforGreeks
* https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
2. Example of Thresholding technique by OpenCV
* https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
3. Thresholding technique by OpenCV
* https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59
4. OpenCV Python Tutorial For Beginners 15 - Adaptive Thresholding
* https://youtu.be/Zf1F4cz8GHU?si=GEn4ItCy7pPwpT0P

'''
