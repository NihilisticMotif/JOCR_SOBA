from PIL import Image 
from ImageUtility import OddKernelArea
import numpy as np 
import cv2
from GrayImage import GrayImage

########################################################################################################################################################

def BinaryPx(img,threshold_px,max_px=255,isinverse=False):
    img=GrayImage(img)
    if isinverse==False:
        # if px > threshold_px then px = max_px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_BINARY)[1]
    else:
        # if px < threshold_px then px = max_px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_BINARY_INV)[1]

def AdaptiveBinaryPx01(img,kernel_area=11,C=2,max_px=255,isinverse=False):
    # Calculating the mean value of the surround pixels within the square kernel_area (width=kernel_area) and use that mean value as the threshold_px
    img=GrayImage(img)
    kernel_area=OddKernelArea(kernel_area)
    if isinverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,kernel_area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,kernel_area,C)

def AdaptiveBinaryPx02(img,kernel_area=11,C=2,max_px=255,isinverse=False):
    # Calculating the "Gaussian" value of the surround pixels within the square kernel_area (width=kernel_area) and use that mean value as the threshold_px
    img=GrayImage(img)
    kernel_area=OddKernelArea(kernel_area)
    if isinverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,kernel_area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,kernel_area,C)

def OtsuBinaryPx(img,max_px=255,isinverse=False):
    # Consider an image with only two distinct image values (bimodal image), 
    # where the histogram would only consist of two peaks. 
    # A good threshold would be in the middle of those two values. 
    # Similarly, Otsu's method determines an optimal global threshold value from the image histogram.
    
    # Limitation of Otsu Method
    # 1. If object kernel_area is much smaller compared to background kernel_area
    # 2. Image is very noisy
    # 3. Image contains kernel_area with different discrete intensities
    # https://youtu.be/jUUkMaNuHP8?si=QnxBvTdVhQW3VTqR

    img=GrayImage(img)
    if isinverse==False:
        return cv2.threshold(img,0,max_px,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    else:
        return cv2.threshold(img,0,max_px,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

########################################################################################################################################################

def TruncatedPx(img,threshold_px,max_px=255):
    img=GrayImage(img)
    # if px > threshold_px then px = threshold_px else px = px
    return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_TRUNC)[1]

def AdaptiveTruncatedPx01(img,kernel_area,C=2,max_px=255):
    img=GrayImage(img)
    kernel_area=OddKernelArea(kernel_area)
    return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_TRUNC,kernel_area,C)

def AdaptiveTruncatedPx02(img,kernel_area,C=2,max_px=255):
    img=GrayImage(img)
    kernel_area=OddKernelArea(kernel_area)
    return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_TRUNC,kernel_area,C)

def OtsuTruncatedx(img,max_px=255,isinverse=False):
    img=GrayImage(img)
    return cv2.threshold(img,0,max_px,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)[1]

########################################################################################################################################################

def Set0Px(img,threshold_px,max_px=255,isinverse=False):
    img=GrayImage(img)
    if isinverse==False:
        # if px > threshold_px then px = px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_TOZERO)[1]
    else:
        # if px < threshold_px then px = px else px = 0
        return cv2.threshold(img,threshold_px,max_px,cv2.THRESH_TOZERO_INV)[1]

def AdaptiveSet0Px01(img,kernel_area,C=2,max_px=255,isinverse=False):
    img=GrayImage(img)
    kernel_area=OddKernelArea(kernel_area)
    if isinverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_TOZERO,kernel_area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_TOZERO_INV,kernel_area,C)

def AdaptiveSet0Px02(img,kernel_area,C=2,max_px=255,isinverse=False):
    img=GrayImage(img)
    kernel_area=OddKernelArea(kernel_area)
    if isinverse==False:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_TOZERO,kernel_area,C)
    else:
        return cv2.adaptiveThreshold(img,max_px,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_TOZERO_INV,kernel_area,C)

def OtsuSet0Px(img,max_px=255,isinverse=False):
    img=GrayImage(img)
    if isinverse==False:
        return cv2.threshold(img,0,max_px,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)[1]
    else:
        return cv2.threshold(img,0,max_px,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)[1]

########################################################################################################################################################

'''
Reference
1. Thresholding technique by GreeksforGreeks
* https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
2. Example of Thresholding technique by OpenCV
* https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
3. Thresholding technique by OpenCV
* https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59
4. OpenCV Python Tutorial For Beginners 15 - Adaptive Thresholding
* https://youtu.be/Zf1F4cz8GHU?si=GEn4ItCy7pPwpT0P
5. Otsu's Method (YouTube Video by Jian Wei Tay)
* https://youtu.be/jUUkMaNuHP8?si=wz5AUX6MYL4Gij5W
'''
