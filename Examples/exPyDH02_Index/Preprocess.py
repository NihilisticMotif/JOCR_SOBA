import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing_V01'
sys.path.insert(1, ImageProcessingPath)
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import ShowImage as show
import Threshold as thresh
import RemoveNoise
from FontSize import ThickFont
import Convolution as con
from Rotation import Rotate
from GrayImage import GrayImage
import FFT as fft
from PIL import Image
from Zoom import RemoveBorders,Zoom
import cv2
import Contour as tour
import numpy as np

img_path = '/Users/imac/Desktop/JOCR_SOBA/exPyDH02_Index/OriginalImage/img.jpeg'
img = show.ReadImage(img_path)
img = Zoom(img,1)

show.ShowImage(img)

img = tour.DefaultDilateImage(img,kernel = np.ones((13,3)))
show.ShowImage(img)
show.SaveImage(img,'DefaultDilateImage13x3')

color_img = tour.DrawDilateContours(dilate_img=img,is_show=True)
show.SaveImage(color_img,'ColorDefaultDilateImage13x3')

'''
python3 Preprocess.py
'''