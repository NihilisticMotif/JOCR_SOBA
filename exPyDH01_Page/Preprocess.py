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
from Contour import DrawDefaultContours

img_path = [
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img_o.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img_r.jpeg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/OriginalJojoSoba.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/BinaryPx_210.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/jojomeme.jpg'
]
img = show.ReadImage(img_path[0])
##img = fft.ReturnFFT(img)
##img = fft.FFTRectangleSharpen(img,10,10)
##fft.SaveFFT(img,'JojoFourier',is_editfft=True)

img = Zoom(img,1.23)
img = GrayImage(img)

img = Rotate(img,is_show=True)
show.SaveImage(img,'TestColor02','TestColor')
print('DrawDefaultContours Activated')
img = DrawDefaultContours(img,is_original_img=False)
show.SaveImage(img,'DrawDefaultContours02','TestColor')

'''
python3 Preprocess.py 
'''