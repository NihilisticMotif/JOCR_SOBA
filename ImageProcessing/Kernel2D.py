import numpy as np 
import cv2
from ImageUtility import OddKernelArea, GetDefaultOption


message = '''
input_options = cv2.MORPH_RECT
 * cv2.MORPH_RECT
 * cv2.MORPH_ELLIPSE,
 * cv2.MORPH_CROSS

# Rectangular Kernel
>>> cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)

# Elliptical Kernel
>>> cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)

# Cross-shaped Kernel
>>> cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)

Reference
* https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
'''

input_options = [
   cv2.MORPH_RECT,
   cv2.MORPH_ELLIPSE,
   cv2.MORPH_CROSS
]

def Kernel2D(width:int, height:int|None = None, scalar:float = 1, mode:int = cv2.MORPH_RECT):
    mode = GetDefaultOption(mode,input_options,message)
    width = width
    if type(height) == int:
        height = height
    else:
        height = width
    return scalar * cv2.getStructuringElement(
	        mode, 
	        (width, height)
        )

def SharpKernel2D(ls:list[float]=[-0.1,-5], center_px:None|float=None):
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
        p = kernel_area - q - 1
        kernel[i][q] = ls[q]
        kernel[i][p] = ls[q]
        kernel[j][q] = ls[q]
        kernel[j][p] = ls[q]

    for i in range(len(ls)):
      j = kernel_area - i - 1
      kernel[len(ls)][i] = ls[i]
      kernel[len(ls)][j] = ls[i]
    
    if not isinstance(center_px, (int, float)):
        center_coef = 1
        center_px = 0
        for i in range(len(ls)):
          j = len(ls) - i - 1
          center_coef += 2
          center_px += (center_coef * 2 + (center_coef-2) * 2) * ls[j]
        center_px *= (-1)
        center_px += 1
    kernel[len(ls)][len(ls)]=center_px
    return kernel 