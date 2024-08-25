from PIL import Image, ImageStat
import numpy as np
import cv2

def IsGray(img):
    # https://youtu.be/jB4hcxdw0Lg?si=zjhbfefsFRRpOrl-
    # https://youtu.be/j-01cO73qRU?si=-ldaSdr217k9dj_1
    if len(img.shape) == 2:
        return True
    elif len(img.shape) == 3:
        return False
    else:
        # https://youtu.be/mI9FIugGIZQ?si=hoC_FHJpn5zA83Kv
        print('Absolute Genderless (Gonadal dysgenesis) : 1 in 150,000')
        print('img has Invalid Shape')
        print('Reported by ImageProcessing / EditImage.py / def IsGray(img)')
        return None

def GrayImage(img):
    if IsGray(img):
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def InvertedImage(img):
    return cv2.bitwise_not(img)

def ColorImage(img):
    if IsGray(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        return img


