import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessin'
sys.path.insert(1, ImageProcessingPath)

import Image
import cv2 
import numpy as np
from Blur import MeanBlur, GaussBlur, BilateralBlur
from Morphology import ThinFont, ThickFont, RemoveNoise, Dilate, Erode, Opening, Canny
from Threshold import DescribeThreshold, Threshold, AdaptiveThreshold
from Kernel2D import Kernel2D, SharpKernel2D
from FFT2D import FFTBlur, FFTSharp

class GRAYImage(Image):
    def __init__(self):
        super(Image, self).__init__()
        if type(self["img"]) == str:
            self.img = cv2.imread(self["img"])
        elif len(self["img"].shape)==3:
            self.img = cv2.cvtColor(self["img"],cv2.COLOR_BGR2GRAY)
        elif len(self["img"].shape)==2:
            self.img = np.copy(self["img"])

########################################################################################################################################################
    # Morphology.py

    def RemoveNoice(self):
        self.img = RemoveNoise(self.img)

    def ThinFont(self):
        self.img = ThinFont(self.img)

    def ThickFont(self):
        self.img = ThickFont(self.img)

    def Dilate(self):
        self.img = Dilate(self.img)

    def Erode(self):
        self.img = Erode(self.img)

    def Opening(self):
        self.img = Opening(self.img)

    def Canny(self):
        self.img = Canny(self.img)

########################################################################################################################################################
    # Threshold.py 

    def Threshold(self, method = cv2.THRESH_BINARY, threshold_px=None, max_px=255):
        transformation = Threshold(method,threshold_px,max_px)
        self.img = transformation.Edit(self.img)

    def AdaptiveThreshold(self,method,adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C, kernel_area=11,constant=2,max_px=255):
        transformation = AdaptiveThreshold(method, adaptive_method, kernel_area, constant, max_px)
        self.img = transformation.Edit(self.img)

########################################################################################################################################################
    # Kernel2D.py

    def SharpFilter2D(self, ls=[-0.1,-5],center_px=None):
        kernel2d = SharpKernel2D(ls, center_px)
        self.img = cv2.filter2D(self.img, -1, kernel2d)

########################################################################################################################################################
    # Blur.py 

    def MeanBlur(self,kernel_area=15,scalar=None):
        self.img = MeanBlur(self.img,kernel_area,scalar)

    def GaussBlur(self,kernel_area=15):
        self.img = GaussBlur(self.img,kernel_area)

    def BilateralBlur(self,kernel_area=15,effect=75):
        self.img = BilateralBlur(self.img,kernel_area,effect)

########################################################################################################################################################
    # FFT2D.py 
    
    def FFTBlur(self, updated_rows, updated_cols, new_frequency = 0, mode = cv2.MORPH_RECT):
        self.img = FFTBlur(self.img, updated_rows, updated_cols, new_frequency = new_frequency, mode = mode)

    def FFTSharp(self, updated_rows, updated_cols, new_frequency = 0, mode = cv2.MORPH_RECT):
        self.img = FFTSharp(self.img, updated_rows, updated_cols, new_frequency = new_frequency, mode = mode)

########################################################################################################################################################


