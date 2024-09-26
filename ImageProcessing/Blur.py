import cv2 
import numpy as np
from ImageUtility import OddKernelArea

def MeanBlur(img:np.ndarray, kernel_area:int = 15, scalar:None|int|float = None):
    # Update each pixel value to average pixel value with in the kernel_area to Blur image
    kernel_area = OddKernelArea(kernel_area)
    if scalar == None:
        scalar = (1/kernel_area)**2
    kernel = scalar*np.ones((kernel_area,kernel_area))
    # https://youtu.be/KuXjwB4LzSA?si=mt-leKGKjpMnJGfg
    # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    return cv2.filter2D(img, -1, kernel)

def GaussBlur(img:np.ndarray, kernel_area:int = 15):
    # Blur the image based in the pixel with in kernel_area using Gaussian function
    # https://www.geeksforgeeks.org/python-image-blurring-using-opencv/
    kernel_area = OddKernelArea(kernel_area)
    return cv2.GaussianBlur(img,(kernel_area,kernel_area), 0) 

def BilateralBlur(img:np.ndarray, kernel_area:int = 15, effect:int = 75):
    # Remove the noise and preserve the edge.
    # https://youtu.be/LjbYKWAQA5s?si=1br6Rl9OYkTZQPJB
    # https://www.tutorialspoint.com/opencv/opencv_bilateral_filter.htm
    kernel_area = OddKernelArea(kernel_area)
    return cv2.bilateralFilter(img, kernel_area, effect, effect)