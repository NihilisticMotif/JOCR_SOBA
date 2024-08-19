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

def Zoom(img, zoom=1, angle=0, coord=None):
    # https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
    # zoom < 1 implies Zoom out
    # zoom > 1 implies Zoom in
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    rotation_matrix = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rotation_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)    
    return result

def CreateBorders(img,size=50,color = [255, 255, 255],IsGray=True):
    top, bottom, left, right = [size]*4
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if IsGray==True:
        img = GrayImage(img)
        return img 
    else:
        return img 
