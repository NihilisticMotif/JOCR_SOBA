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


img_path = [
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img_o.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img_r.jpeg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/OriginalJojoSoba.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/BinaryPx_210.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/jojomeme.jpg'
]
img_o = show.ReadImage(img_path[0])
img = Zoom(img_o,1.23)
img = Rotate(img)
img = GrayImage(img)
#show.ShowImage(img)

ost_img = thresh.OtsuSet0Px(img)
ost_img = thresh.BinaryPx(img,200)
#show.SaveImage(ost_img,'ost_img')

#fft_img = fft.FFTSharpen01(img,10,10)
#show.SaveImage(fft_img,'fft_img')

thick_img = ThickFont(ost_img)
thick_img = ThickFont(thick_img)
show.ShowImage(thick_img)

#fft_img = fft.FFTBlur03(ost_img,10,10)
#show.ShowImage(fft_img)

#erode_img = con.Sharpen(img)
#erode_img = ThickFont(erode_img)
#show.ShowImage(erode_img)

'''
python3 Preprocess.py 
'''