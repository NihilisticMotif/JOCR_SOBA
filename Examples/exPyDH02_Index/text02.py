import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)

from ColumnSegmentation import ColumnSegmentation 
import ShowImage as show 

img_path = '/Users/imac/Desktop/JOCR_SOBA/exPyDH02_Index/OriginalImage/img.jpeg'
img = show.ReadImage(img_path)
ColumnSegmentation(img,width=20,height=200,is_show=True,img_title='ColumnSegmentation',is_multiple_imgs=True)
'''
python3 text02.py
'''