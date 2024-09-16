import numpy as np 
import cv2
from ImageFunction import OddKernelArea

class Kernel2D:
    def __init__(
            self,
            width:int,
            height:None|int,
            scalar:float = 1,
            type = cv2.MORPH_RECT):
        self.width = OddKernelArea(width)
        if height == None:
            self.height = OddKernelArea(width) 
        else:
            self.height = OddKernelArea(height) 
        self.scalar = scalar 
        self.type = type 
        self._kernel = self.scalar * cv2.getStructuringElement(self.type, (self.width, self.height))

    def Help():
        message = '''
Kernel2D is used for 
 * edge detection
 * blur image
 * convolution 
 * FFT convolution

type Mode
1. cv2.MORPH_RECT

[[1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]]

2. cv2.MORPH_ELLIPSE

[[0 0 0 1 1 1 1 1 0 0 0]
 [0 1 1 1 1 1 1 1 1 1 0]
 [1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1]
 [0 1 1 1 1 1 1 1 1 1 0]]

3. cv2.MORPH_CROSS

[[0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0]]

Reference
* https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
''' 
        print(message)

    @property
    def UpdateKernel(self):
        # https://www.pythonmorsels.com/making-read-only-attribute/
        # https://www.reddit.com/r/learnpython/comments/twvwt8/what_is_best_practice_for_updating_class_attr/
        self._kernel = self.scalar * cv2.getStructuringElement(self.type, (self.width, self.height))
    


