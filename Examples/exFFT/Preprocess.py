import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import ShowImage as show
import FFT as fft
from GrayImage import GrayImage, ColorImage
import cv2

img_path = '/Users/imac/Desktop/JOCR_SOBA/exFFT/Image/jojomeme.jpg'
img = show.ReadImage(img_path)
img = GrayImage(img)
fft_img = fft.FFTSharpen02(img,20,10,is_show=True)

show.SaveImage(fft_img,'fft_img')
fft.SaveFFT(fft_img,'sharp')
fft.SaveFFT(show.ReadImage(img_path),'original')

'''
python3 Preprocess.py
'''