import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import ShowImage as show
import Threshold
import RemoveNoise
import FontSize
import Convolution
from Rotation import Rotate
from GrayImage import GrayImage
import FFT as fft
from PIL import Image
from Zoom import RemoveBorders,Zoom
import cv2

img_path = [
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img_o.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img_r.jpeg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/OriginalJojoSoba.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/BinaryPx_210.jpg'
]
img = show.ReadImage(img_path[3])
#img=Zoom(img,zoom=1.23)
'''
show.ShowImage(img)
img=GrayImage(img)
img=RemoveBorders(img)
img=Rotate(img,threshold_px=200)'''
show.ShowImage(img)
#img = fft.FFTRSharpen(img,1,1)
img = fft.FFTRBlur(img,2,2,is_show=True)
#show.ShowImage(img)

#show.SaveImage(img,"fft01_FFTRSharpen__img_10_10",folder='FFT')

'''
python3 Preprocess.py 
'''