import numpy as np
from PIL import Image 
import PIL

def Numpy2Image(img):
    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    if type(img)==np.ndarray:
        img=Image.fromarray(img)
    return img

def Image2Numpy(img):
    return np.array(img)

'''
Reference
1. Tesseract OCR Tutorial
* https://nanonets.com/blog/ocr-with-tesseract/
2. How to Open an Image in Python with PIL
* https://youtu.be/UxYJxcdLrs0?si=biuELY46TWTwhqvX

'''