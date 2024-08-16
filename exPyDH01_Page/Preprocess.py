import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

import ShowImage as show
import Threshold
import RemoveNoise
import FontSize
import Border 
import Convolution
from Rotation import Rotate
from GrayImage import GrayImage
from PIL import Image
import cv2

img_path01 = '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/Image/image_Original.jpg'
img_path02 = '/Users/imac/Desktop/JOCR_SOBA/exJojoSoba/IMG_7563.jpeg'
img_path03 = '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/Image/page_01_rotated.jpeg'
img = show.ReadImage(img_path03)
img=GrayImage(img)
img=Rotate(img)
Index=''
#img=RemoveNoise.RemoveNoise(img)
#img=Threshold.BinaryPx(img,210)

#img=Convolution.Sharpen(img)
#img=Convolution.Sharpen(img)
#img=FontSize.ThickFont(img)
Threshold.BinaryPx(img,200)#,IsInverse=True)
#img=Border.CreateBorders(img,color=[255,0,0],IsGray=False)
show.ShowImage(img)
show.SaveImage(img,"Sharpen_New"+str(Index))

'''
python3 Preprocess.py 
'''