import numpy as np 
import cv2 

def OddKernelArea(num):
    num=int(num)
    if num<3:
        return 3 
    else:
        if num%2==1:
            return num 
        else:
            return num+1

def Px255(num):
    if num < 0:
        return 0
    elif num > 255:
        return 255
    else:
        return num

def Sharpen(ls:list[float]=[-0.1,-5],center_px:None|float=None):
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
    return kernel
