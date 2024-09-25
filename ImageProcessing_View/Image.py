import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)

from ShowImage import Show, Save
from Zoom import RemoveBorders, Zoom, CreateBorders, Crop
import cv2
import numpy as np 
import GrayImage
from Rotate import Rotate

class Image:
    def __init__(
            self, 
            img:np.ndarray | str
            ):
        if type(img) == str:
            self.img = cv2.imread(img)
        elif len(img.shape)==3:
            self.img = np.copy(img)
        elif len(img.shape)==2:
            self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            print('Error: the input img is invalid.')
            print('len(img.shape) in [2, 3] or type(img) == str should be true.')
            print('Reported by ImageProcessing_Biew / GrayImage.py / class Image / def __init__(self, img)')

########################################################################################################################################################
    # Get Data

    def GetImg(self):
        return self.img

    def GetGrayImg(self):
        return GrayImage(self.img)

    def GetColorImg(self):
        return Image(self.img)

    def Show(self, title:str = 'Image'):
        Show(self.img, title)

    def Save(self, img_title:str = 'Image', folder:str = 'Image', fileformat:str = 'jpg'):
        Save(self.img, img_title, folder, fileformat)

    def shape(self):
        return self.img.shape

########################################################################################################################################################
    # Edit Data

    def Zoom(self, zoom:int = 1):
        self.img = Zoom(self.img, zoom)

    def RemoveBorders(self):
        self.img = RemoveBorders(self.img)

    def Crop(self, x:int|None = None, y:int|None = None, width:int|None = None, height:int|None = None):
        self.img = Crop(self.img, x, y, width, height)

    def CreateBorders(self, size:int = 50):
        self.img = CreateBorders(self.img, size)

    def Rotate(self, angle:int|None = None):
        self.img = Rotate(self.img, angle)


