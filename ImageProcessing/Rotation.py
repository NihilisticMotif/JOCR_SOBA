from PIL import Image 
from Utility import Numpy2Image, Image2Numpy
from GrayImage import GrayImage
from Convolution import GaussBlur
from Threshold import OtsuBinaryPx
import numpy as np 
import cv2

def GetSkewAngle(img):
    # https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb
    
    #####################################################################################################################
    
    newImage = img.copy()
    gray = GrayImage(newImage)
    blur = GaussBlur(gray,9)
    thresh = OtsuBinaryPx(blur,isinverse=True)
    
    #####################################################################################################################
    
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    kernel = np.ones((5,30))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    #####################################################################################################################

    # Find all contours
    '''
    Read this
    * https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

    Then fix this code
    '''
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)
    print('What is contours ?')
    print(type(contours))
    print('Show elements of contours.')
    for i in contours:
        print(i)
    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print('largestContour')
    print(largestContour)
    #print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def Rotate(img, angle=None):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    if angle==None:
        angle = GetSkewAngle(img)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, 
        rotation_matrix, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
        )


