import cv2 
from GrayImage import GrayImage
# https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb

def RemoveBorders(img):
    contours, heiarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]
    return (crop)

def CreateBorders(img,size=50,color = [255, 255, 255],IsGray=True):
    top, bottom, left, right = [size]*4
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if IsGray==True:
        img = GrayImage(img)
        return img 
    else:
        return img 
