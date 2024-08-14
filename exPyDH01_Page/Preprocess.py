import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import ShowImage as show
import EditImage as edit
import Threshold
from PIL import Image
import cv2

img_path = '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/Image/image_Original.jpg'
img = show.ReadImage(img_path)
Index=''
img=Threshold.SetPx_as_Zero(img,200)
show.ShowImage(img)
show.SaveImage(img,"SetPx_as_Zero"+str(Index))

'''
python3 preprocess.py
'''