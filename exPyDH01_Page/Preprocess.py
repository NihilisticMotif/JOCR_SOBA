import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
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


img_path = '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img.jpg'
img_o = show.ReadImage(img_path)
img = Zoom(img_o,1.23)
img = Rotate(img,is_show=True)
img = GrayImage(img)

ost_img = thresh.OtsuBinaryPx(img)
show.SaveImage(ost_img,'OtsuBinaryPx')

thick_img = ThickFont(ost_img)
show.SaveImage(thick_img,'ThickFont')

'''
python3 Preprocess.py 
'''