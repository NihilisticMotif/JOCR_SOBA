from PIL import Image 
from Utility import Numpy2Image, Image2Numpy
from GrayImage import GrayImage, ColorImage
from Convolution import GaussBlur
from Threshold import OtsuBinaryPx,BinaryPx
from ShowImage import ShowImage, SaveImage
import numpy as np 
import cv2

def GetSkewAngle(img,threshold_px,is_binary_inv=False,dilate_img=None):
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
        dilate_img = GrayImage(new_img)
        dilate_img = GaussBlur(dilate_img,9)
        if isinstance(threshold_px, (int,float)):
            dilate_img = BinaryPx(dilate_img,threshold_px,isinverse=is_binary_inv)   
        dilate_img = OtsuBinaryPx(dilate_img,isinverse=True)   
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        kernel = np.ones((2,30))
        dilate_img = cv2.dilate(dilate_img, kernel, iterations=2)

    #####################################################################################################################

    # 3.rd Find text blocks (contour detect white area)

    # 4.th There can be various approaches to determine skew angle, 
    # but we’ll stick to the simple one — take the largest text block and use its angle.

    contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    largestContour = contours[0]

    #####################################################################################################################

    # 5.th calculating angle.
    # The angle value always lies between [-90,0).
    # https://theailearner.com/tag/cv2-minarearect/

    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    
    '''    
    #####################################################################################################################
    
    # This is for showing the contours detection.
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    print('type(new_img)',type(new_img))
    original_angle = minAreaRect[-1]
    Folder = 'RotateImage'+'Zoom_o'
    print('angle______________ =',angle)
    print('original_angle_____ =',original_angle)
    print('len(contours)______ = ',len(contours))
    print('len(largestContour) = ',len(largestContour))
    ShowImage(dilate_img,'dilate')
    SaveImage(dilate_img,'dilate_img',Folder)
    
    cnew_img = ColorImage(new_img)
    cnew_img0 = cv2.drawContours(cnew_img, largestContour, -1, (0,0,255), 3)
    ShowImage(cnew_img0,'image largestContour')
    cnew_img = cv2.drawContours(cnew_img, contours, -1, (0,0,255), 3)
    ShowImage(cnew_img, 'image Contours')

    cdilate_img = ColorImage(dilate_img)
    cdilate_img0 = cv2.drawContours(cdilate_img, largestContour, -1, (0,0,255), 3)
    ShowImage(cdilate_img0,'dilate largestContour')
    cdilate_img = cv2.drawContours(cdilate_img, contours, -1, (0,0,255), 3)
    ShowImage(cdilate_img,'dilate Contours')
    SaveImage(cdilate_img,'cdilate_img',Folder)
    
    #####################################################################################################################
    #'''

    return angle

def Rotate(img, angle=None,threshold_px=None,is_binary_inv=False,dilate_img=None):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    if not isinstance(angle,(int,float)):
        if not isinstance(dilate_img,np.ndarray):
            angle = GetSkewAngle(img,threshold_px=threshold_px,is_binary_inv=is_binary_inv)
        else:
            angle = GetSkewAngle(img,threshold_px=threshold_px,is_binary_inv=is_binary_inv,dilate_img=dilate_img)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, 
        rotation_matrix, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
        )


