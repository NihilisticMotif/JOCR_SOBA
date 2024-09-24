import cv2 
from ImageUtility import SetPx, ReturnValidInput, OddKernelArea

message1='''
Basic Parameter
1. max_px = 255
2. method = cv2.THRESH_BINARY
* cv2.THRESH_BINARY         
* cv2.THRESH_BINARY_INV
* cv2.THRESH_TRUNC
* cv2.THRESH_TOZERO
* cv2.THRESH_TOZERO_INV

Threshold
1. threshold_px
* if type(threshold_px) == int : activating customized threshold
* if type(threshold_px) != int : activating Otsu threshold

AdaptiveThreshold
1. kernel_area = 11
2. constant = 2
3. additional_method = cv2.ADAPTIVE_THRESH_MEAN_C
* cv2.ADAPTIVE_THRESH_MEAN_C
* cv2.ADAPTIVE_THRESH_GAUSSIAN_C

Reference
* https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576ac7e89a5e95490116e7d2082b3096b2b8
'''

message2='''
cv2.THRESH_BINARY
# if px > threshold_px then px = max_px else px = 0

cv2.THRESH_BINARY_INV
# if px < threshold_px then px = max_px else px = 0

cv2.THRESH_TRUNC
# if px > threshold_px then px = threshold_px else px = px

cv2.THRESH_TOZERO
# if px > threshold_px then px = px else px = 0

cv2.THRESH_TOZERO_INV
# if px < threshold_px then px = px else px = 0

cv2.ADAPTIVE_THRESH_MEAN_C
# Calculating the mean value of the surround pixels within the square kernel_area (width=kernel_area) and use that mean value as the threshold_px

cv2.ADAPTIVE_THRESH_GAUSSIAN_C
# Calculating the "Gaussian" value of the surround pixels within the square kernel_area (width=kernel_area) and use that mean value as the threshold_px

cv2.THRESH_OTSU
# Consider an image with only two distinct image values (bimodal image), 
# where the histogram would only consist of two peaks. 
# A good threshold would be in the middle of those two values. 
# Similarly, Otsu's method determines an optimal global threshold value from the image histogram.
# Limitation of Otsu Method
# 1. If object kernel_area is much smaller compared to background kernel_area
# 2. Image is very noisy
# 3. Image contains kernel_area with different discrete intensities
# https://youtu.be/jUUkMaNuHP8?si=QnxBvTdVhQW3VTqR

Reference
* https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576ac7e89a5e95490116e7d2082b3096b2b8
'''

new_line = '########################################################################################################################################################'

source = 'Reported by ImageProcessing / Threshold.py /'

long_message = new_line+'\n'+message1+'\n'+new_line+'\n'+message2+'\n'+new_line+'\n'+source
short_message = new_line+'\n'+message1+'\n'+new_line+'\n'

method_options =  [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

class Threshold:
    def __init__(self, method = cv2.THRESH_BINARY, threshold_px=None, max_px=255):
        self.method = ReturnValidInput(method, method_options, short_message)
        if type(threshold_px) == int: 
            self.threshold_px = SetPx(threshold_px)
        else:
            self.threshold_px = None
        self.max_px = SetPx(max_px) 

    def Edit(self,img):
        if type(self.threshold_px) == int:
            return cv2.threshold(img, self.threshold_px, self.max_px, self.method)[1]
        else:
            return cv2.threshold(img, 0, self.max_px, self.method + cv2.THRESH_OTSU)[1]

class AdaptiveThreshold:
    def __init__(self, method = cv2.THRESH_BINARY, adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C, kernel_area=11, constant=2,max_px=255):
        self.method = ReturnValidInput(method, method_options, short_message)
        self.kernel_area = OddKernelArea(kernel_area)
        self.constant = constant
        self.max_px = SetPx(max_px)
        self.adaptive_method = ReturnValidInput(adaptive_method, [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C], short_message)
    
    def Edit(self,img):
        return cv2.adaptiveThreshold(img, self.max_px, self.adaptive_method, self.method, self.kernel_area, self.constant)




