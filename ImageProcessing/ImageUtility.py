import numpy as np
from PIL import Image 

def Something():
    print('ChatGPT is stupid.')

def Numpy2Image(img):
    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    if type(img)==np.ndarray:
        img=Image.fromarray(img)
    return img

def Image2Numpy(img):
    return np.array(img)

def OddKernelArea(num):
    num=int(num)
    if num<3:
        return 3 
    else:
        if num%2==1:
            return num 
        else:
            return num+1

'''
Reference
1. Tesseract OCR Tutorial
* https://nanonets.com/blog/ocr-with-tesseract/
2. How to Open an Image in Python with PIL
* https://youtu.be/UxYJxcdLrs0?si=biuELY46TWTwhqvX

'''