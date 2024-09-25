import cv2
import numpy as np

def InvertedImage(img:np.ndarray):
    return cv2.bitwise_not(img)

def OddKernelArea(num:int):
    num=int(num)
    if num<3:
        return 3 
    else:
        if num%2==1:
            return num 
        else:
            return num+1

def SetPx(num:int):
    num=int(num)
    if num < 0:
        return 0 
    elif num > 255:
        return 255
    else:
        return num

def GetDefaultOption(input:any, input_options:list, message:str):
    if input not in input_options:
        print(message)
        return input_options[0]
    else:
        return input

def GetSize(size:int|None, max_size:int|None = None, default_size:int = 0):
    if type(size) == int:
        if max_size == None:
            max_size = size
        if size < 0:
            return 0
        elif size > max_size:
            return max_size
        else:
            return size
    else:
        return default_size

'''
Reference
1. Tesseract OCR Tutorial
* https://nanonets.com/blog/ocr-with-tesseract/
2. How to Open an Image in Python with PIL
* https://youtu.be/UxYJxcdLrs0?si=biuELY46TWTwhqvX

'''