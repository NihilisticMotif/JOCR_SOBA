import cv2
import numpy as np
from PIL import Image 
import os
from Utility import Numpy2Image, Image2Numpy
from GrayImage import IsGray

def ReadImage(img_path):
    return cv2.imread(img_path)

def ShowImage(img,title='image'):
    img=Image2Numpy(img)
    if IsGray(img)==False:
        # https://www.geeksforgeeks.org/convert-bgr-and-rgb-with-python-opencv/
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def DescribeImage(img_path,detail=None):
    if type(detail)==str:
        if detail.lower() == 'size':
            return Image.open(img_path).size
        elif detail.lower() == 'mode':
            return Image.open(img_path).mode
        elif detail.lower() == 'type':
            return type(Image.open(img_path))
        else:
            return Image.open(img_path)
    else:
        return Image.open(img_path)

def SaveImage(img,img_title,folder='Image',fileformat='.jpg'):
    # https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
    if not os.path.exists(folder):
        os.makedirs(folder)
    img_path = os.path.join(folder,img_title+fileformat)
    img = Numpy2Image(img)
    img.save(img_path)

'''
Reference
1. How to Open an Image in Python with PIL (Pillow) (OCR in Python 02.01)
* https://youtu.be/UxYJxcdLrs0?si=rrV2I7AnYIAI6Q2f

'''