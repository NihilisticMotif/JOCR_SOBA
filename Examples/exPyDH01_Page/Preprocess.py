import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing_View'
sys.path.insert(1, ImageProcessingPath)
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

from GrayImage import GrayImage
import cv2

path = '/Users/imac/Desktop/JOCR_SOBA/Examples/exPyDH01_Page/OriginalImage/img.jpg'

img = GrayImage(path)
img.Zoom(-1.23)
img.Rotate()
img.Threshold()
img.Show()

'''
python3 Preprocess.py 
'''