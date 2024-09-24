import cv2
import numpy as np
from PIL import Image 
import os

def Show(img,title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Save(img,img_title='Image',folder='Image',fileformat='jpg'):
    # https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
    if not os.path.exists(folder):
        os.makedirs(folder)
    if fileformat[0]=='.':
        fileformat = fileformat[1:]
    img_path = os.path.join(folder,img_title+'.'+fileformat)
    img.save(img_path)
