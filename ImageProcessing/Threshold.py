from ImageFunction import Px255, OddKernelArea
import cv2 
class Threshold:
    def __init__(self,
                 method=cv2.THRESH_BINARY,
                 threshold_px=cv2.THRESH_OTSU,
                 max_px=255,
                 kernel_area=15,
                 constant=2
                 ):
        # https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
        self.method = method
        self.threshold_px = threshold_px
        self.max_px = Px255(max_px)
        self.kernel_area = OddKernelArea(kernel_area)
        self.constant = constant

    def Help():
        message = '''
Threshold is used for image processing.

method Modes
1. cv2.THRESH_BINARY
 * if px > threshold_px then px = max_px else px = 0
2. cv2.THRESH_BINARY_INV
 * if px < threshold_px then px = max_px else px = 0
3. cv2.THRESH_TOZERO
 * if px > threshold_px then px = px else px = 0
4. cv2.THRESH_TOZERO_INV
 * if px < threshold_px then px = px else px = 0
5. cv2.THRESH_TRUNC
 * if px > threshold_px then px = threshold_px else px = px

threshold_px Modes
1. type(threshold_px) == int 
 * Select the threshold pixel between 0 - 255 Manually
2. cv2.ADAPTIVE_THRESH_MEAN_C
 * Calculating the mean value of the surround pixels within the square kernel_area 
 * (width=kernel_area) and use that mean value as the threshold_px
3. cv2.ADAPTIVE_THRESH_GAUSSIAN_C
 * Calculating the "Gaussian" value of the surround pixels within the square kernel_area 
 * (width=kernel_area) and use that mean value as the threshold_px
4. cv2.THRESH_OTS
 * Consider an image with only two distinct image values (bimodal image), 
 * where the histogram would only consist of two peaks. 
 * A good threshold would be in the middle of those two values. 
 * Similarly, Otsu's method determines an optimal global threshold value from the image histogram.
 * Limitation of Otsu Method
 * 1. If object kernel_area is much smaller compared to background kernel_area
 * 2. Image is very noisy
 * 3. Image contains kernel_area with different discrete intensities
   
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
        print(message)