import cv2 
import numpy as np 
from Utility import OddKernelArea

########################################################################################################################################################

def Sharpen(img,ls=[-0.1,-5],center_px=None):
    # https://youtu.be/KuXjwB4LzSA?si=mt-leKGKjpMnJGfg
    # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    # Edge Detection
    # Time : O(n^2)
    # Space: O(n^2)
    kernel_area=len(ls)*2+1
    kernel = np.ones((kernel_area,kernel_area))
    for i in range(len(ls)):
      j = kernel_area-i-1
      kernel[i] = ls[i]*kernel[i]
      kernel[j] = ls[i]*kernel[j]

      for q in range(i):
        p = kernel_area-q-1
        kernel[i][q] = ls[q]
        kernel[i][p] = ls[q]
        kernel[j][q] = ls[q]
        kernel[j][p] = ls[q]

    for i in range(len(ls)):
      j = kernel_area-i-1
      kernel[len(ls)][i] = ls[i]
      kernel[len(ls)][j] = ls[i]
    
    if not isinstance(center_px, (int, float)):
        center_coef = 1
        center_px = 0
        for i in range(len(ls)):
          j = len(ls) - i -1
          center_coef += 2
          center_px += (center_coef * 2 + (center_coef-2) * 2) * ls[j]
        center_px *= (-1)
        center_px += 1
    kernel[len(ls)][len(ls)]=center_px
    return cv2.filter2D(img, -1, kernel)

def MeanBlur(img,kernel_area=15,scalar=None):
    # Update each pixel value to average pixel value with in the kernel_area to Blur image
    kernel_area=OddKernelArea(kernel_area)
    if scalar==None:
        scalar=(1/kernel_area)**2
    kernel = scalar*np.ones((kernel_area,kernel_area))
    # https://youtu.be/KuXjwB4LzSA?si=mt-leKGKjpMnJGfg
    # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    return cv2.filter2D(img, -1, kernel)

def GaussBlur(img,kernel_area=15):
    # Blur the image based in the pixel with in kernel_area using Gaussian function
    # https://www.geeksforgeeks.org/python-image-blurring-using-opencv/
    kernel_area=OddKernelArea(kernel_area)
    return cv2.GaussianBlur(img,(kernel_area,kernel_area), 0) 

def BilateralBlur(img,kernel_area=15,effect=75):
    # Remove the noise and preserve the edge.
    # https://youtu.be/LjbYKWAQA5s?si=1br6Rl9OYkTZQPJB
    # https://www.tutorialspoint.com/opencv/opencv_bilateral_filter.htm
    kernel_area=OddKernelArea(kernel_area)
    return cv2.bilateralFilter(img, kernel_area, effect, effect)

########################################################################################################################################################

def NormalDistribution(x,std=1,mean=0):
    std2=std**2
    coefficient = 1/( (2*np.pi*std2)**(1/2) )
    power = -1*( (x-mean)**2 )/( 2*std2 )
    return coefficient*np.exp(power)

def ChessGaussBlur(img,std=0.1,mean=0.3,kernel_area=3,center_px='None'):
    ls=[]
    for i in range(kernel_area):
        ls.append(NormalDistribution(i,std,mean))
    if center_px.lower() == 'default':
           center_px=center_px
    elif not isinstance(center_px, (int, float)):
        center_px = NormalDistribution(len(ls),std,mean)
    return Sharpen(img,ls,center_px=center_px)
