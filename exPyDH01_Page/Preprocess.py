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
import Contour as tour

img_path = '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img.jpg'
img = show.ReadImage(img_path)
img = Zoom(img,1.23)
img = Rotate(img,is_show=True)
'''
original_angle: 89.46703338623047
output_angle__: -0.5329666137695312
Reported by ImageProcessing / Rotation.py / def GetSkewAngle

rotation_matrix
 [[ 0.99995674 -0.00930189  7.10469492]
 [ 0.00930189  0.99995674 -5.54820932]]
Reported by ImageProcessing / Rotation.py / def Rotate
'''

ost_img = thresh.OtsuBinaryPx(img)
show.SaveImage(ost_img,'OtsuBinaryPx')

thick_img = ThickFont(ost_img)
show.SaveImage(thick_img,'ThickFont')

'''
python3 Preprocess.py 
'''