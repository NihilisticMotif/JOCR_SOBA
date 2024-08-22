from PIL import Image 
from Utility import Numpy2Image, Image2Numpy
from GrayImage import GrayImage, ColorImage
from Convolution import GaussBlur
from Threshold import OtsuBinaryPx,BinaryPx
from ShowImage import ShowImage, SaveImage
import numpy as np 
import cv2
from Contour import DefaultDilateImage, GetLargestContour, GetContours, DrawContours

def GetSkewAngle(img,threshold_px,is_binary_inv=False,dilate_img=None,kernel = np.ones((2,30)),is_show=False):
    # https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb
    # https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df

    #####################################################################################################################
    
    # 1.st Step
    # Convert image to gray image and blur the image to rease noise in the image.

    # 2.nd Step
    # Then find the area with text 
    # With a larger kernel on X axis to get rid of all spaces between words, 
    # and a smaller kernel on Y axis to blend in lines of one block between each other,
    # but keep larger spaces between text blocks intact. (in the original state or similar to that state)
    # text becomes white (255,255,255), and background is black (0,0,0)

    new_img = img.copy()
    if not isinstance(dilate_img,np.ndarray):
        dilate_img = DefaultDilateImage(
            img=new_img,
            threshold_px=threshold_px,
            is_binary_inv=is_binary_inv,
            kernel = kernel)

    #####################################################################################################################

    # 3.rd Find text blocks (contour detect white area)

    # 4.th There can be various approaches to determine skew angle, 
    # but we’ll stick to the simple one — take the largest text block and use its angle.
    largestContour = GetLargestContour(dilate_img)
    contour = GetContours(dilate_img)

    #####################################################################################################################

    # 5.th calculating angle.
    # The angle value always lies between [-90,0).
    # https://theailearner.com/tag/cv2-minarearect/

    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = angle - 90

    #####################################################################################################################
    
    # This is for showing the contours detection.
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    if is_show==True:
        show_img = DrawContours(img,contour)
        ShowImage(show_img,'Show Contour 01')
        show_img = DrawContours(dilate_img,contour)
        ShowImage(show_img,'Show Contour 02')   
        print()
        print('original_angle:',minAreaRect[-1])
        print('output_angle__:',angle)
        print('Reported by ImageProcessing / Rotation.py / def GetSkewAngle')
        print()

    #####################################################################################################################

    return angle

def Rotate(img, angle=None,threshold_px=None,is_binary_inv=False,dilate_img=None,is_show=False):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    if not isinstance(angle,(int,float)):
        if not isinstance(dilate_img,np.ndarray):
            angle = GetSkewAngle(img,threshold_px=threshold_px,is_binary_inv=is_binary_inv,is_show=is_show)
        else:
            angle = GetSkewAngle(img,threshold_px=threshold_px,is_binary_inv=is_binary_inv,dilate_img=dilate_img,is_show=is_show)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, 
        rotation_matrix, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
        )


