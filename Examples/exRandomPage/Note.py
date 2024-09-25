import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import ShowImage as show
import FFT as fft
from GrayImage import GrayImage, ColorImage
import cv2

img_path = '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/RandomImage/jojomeme.jpg'

img = show.ReadImage(img_path)
img = GrayImage(img)
fft_img = fft.FFTSharpen02(img,10,10,is_show=True)

show.SaveImage(fft_img,'fft_img')

'''
python3 Note.py
'''