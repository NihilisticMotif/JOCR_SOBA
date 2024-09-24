import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)

from ShowImage import Show, Save
from Zoom import RemoveBorders, Zoom, CreateBorders, Crop
import cv2
import numpy as np 
import GrayImage

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
            self.img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    def Show(self,title='Image'):
        Show(self.img,title)

    def Save(self,img_title='Image',folder='Image',fileformat='jpg'):
        Save(self.img,img_title,folder,fileformat)

    def Help():
        message='''
        Help Menu
        '''
        print(message)

    def Zoom(self,zoom=1):
        self.img = Zoom(self.img,zoom)

    def RemoveBorders(self):
        self.img = RemoveBorders(self.img)

    def Crop(self, x, y, width, height):
        self.img = Crop(self.img, x, y, width, height)

    def CreateBorders(self,size=50):
        self.img = CreateBorders(self.img,size)

    def Rotate():
        pass 

    def ReturnImg(self):
        return self.img

    def ReturnGrayImg(self):
        return GrayImage(self.img)


