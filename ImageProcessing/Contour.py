import cv2
from GrayImage import ColorImage, GrayImage 
import numpy as np 
from Convolution import GaussBlur
from Threshold import BinaryPx, OtsuBinaryPx
from ShowImage import SaveImage
from ImageUtility import OddKernelArea
import ShowImage as show

#####################################################################################################################

def DefaultDilateImage(
        img,
        threshold_px    =   None,
        kernel          =   np.ones((2,30)),
        kernel_area     =   9,
        is_binary_inv   =   True,
        is_otsu         =   True,
        is_show         =   False):
    dilate_img = GrayImage(img)    
    dilate_img = GaussBlur(dilate_img,OddKernelArea(kernel_area))
    if isinstance(threshold_px, (int,float)):
        dilate_img = BinaryPx(dilate_img,threshold_px,isinverse=is_binary_inv)   
    if is_otsu==True:
        dilate_img = OtsuBinaryPx(dilate_img,isinverse=is_binary_inv)   
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate_img = cv2.dilate(dilate_img, kernel, iterations=2)
    if is_show==True:
        show.ShowImage(dilate_img,'Contour.py')
    return dilate_img

#####################################################################################################################

def GetContours(dilate_img,is_show=False):
    contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if is_show==True:
        DrawContours(dilate_img,contours,rgb=(255,0,0),is_show=is_show)
    return contours

def GetLargestContour(dilate_img,is_show=False):
    contours = GetContours(dilate_img,is_show=is_show)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    return contours[0]

def GetDefaultContours(
        img,
        threshold_px=None,
        kernel = np.ones((2,30)),
        kernel_area=9,
        is_binary_inv=True,
        is_otsu=True,
        is_show=False):
    dilate_img = DefaultDilateImage(
        img,
        threshold_px=threshold_px,
        kernel=kernel,
        kernel_area=kernel_area,
        is_binary_inv=is_binary_inv,
        is_otsu=is_otsu)
    contours = GetContours(dilate_img,is_show=is_show)
    return contours

#####################################################################################################################

def GetRegtangle(contour):
    return cv2.boundingRect(contour)

def SortContours_Left2Right(contour,is_reverse=False):
    return sorted(contour,key=lambda x:cv2.boundingRect(x)[0],reverse=is_reverse)

def SortContours_Top2Bottom(contour,is_reverse=False):
    return sorted(contour,key=lambda x:cv2.boundingRect(x)[1],reverse=is_reverse)

def SortContours_Width(contour,is_reverse=False):
    return sorted(contour,key=lambda x:cv2.boundingRect(x)[2],reverse=is_reverse)

def SortContours_Height(contour,is_reverse=False):
    return sorted(contour,key=lambda x:cv2.boundingRect(x)[3],reverse=is_reverse)

def SortContours_Area(contour,is_reverse=False):
    return sorted(contour,key=lambda x:cv2.contourArea,reverse=is_reverse)

#####################################################################################################################

def GetContours_Left2Right(dilate_img,is_reverse=False):
    # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    # https://www.w3schools.com/python/ref_list_sort.asp
    contour = GetContours(dilate_img)
    contour = SortContours_Left2Right(contour=contour,is_reverse=is_reverse)
    return contour

def GetContours_Top2Bottom(dilate_img,is_reverse=False):
    contour = GetContours(dilate_img)
    contour = SortContours_Top2Bottom(contour=contour,is_reverse=is_reverse)
    return contour

def GetContours_Width(dilate_img,is_reverse=False):
    contour = GetContours(dilate_img)
    contour = SortContours_Width(contour=contour,is_reverse=is_reverse)
    return contour

def GetContours_Height(dilate_img,is_reverse=False):
    contour = GetContours(dilate_img)
    contour = SortContours_Height(contour=contour,is_reverse=is_reverse)
    return contour

def GetContours_Area(dilate_img,is_reverse=False):
    contour = GetContours(dilate_img)
    contour = SortContours_Area(contour=contour,is_reverse=is_reverse)
    return contour 

#####################################################################################################################

def GetDefaultContours_Left2Right(
        img,
        threshold_px    =   None,
        kernel          =   np.ones((2,30)),
        kernel_area     =   9,
        is_binary_inv   =   True,
        is_otsu         =   True,
        is_reverse=False):
    dilate_img = DefaultDilateImage(
            img,
            threshold_px    =   threshold_px,
            kernel          =   kernel,
            kernel_area     =   kernel_area,
            is_binary_inv   =   is_binary_inv,
            is_otsu         =   is_otsu,
            )
    contour = GetContours(dilate_img)
    contour = SortContours_Left2Right(contour=contour,is_reverse=is_reverse)
    return contour

def GetDefaultContours_Top2Bottom(
        img,
        threshold_px    =   None,
        kernel          =   np.ones((2,30)),
        kernel_area     =   9,
        is_binary_inv   =   True,
        is_otsu         =   True,
        is_reverse=False):
    dilate_img = DefaultDilateImage(
            img,
            threshold_px    =   threshold_px,
            kernel          =   kernel,
            kernel_area     =   kernel_area,
            is_binary_inv   =   is_binary_inv,
            is_otsu         =   is_otsu,
            )
    contour = GetContours(dilate_img)
    contour = SortContours_Top2Bottom(contour=contour,is_reverse=is_reverse)
    return contour

def GetDefaultContours_Width(
        img,
        threshold_px    =   None,
        kernel          =   np.ones((2,30)),
        kernel_area     =   9,
        is_binary_inv   =   True,
        is_otsu         =   True,
        is_reverse=False):
    dilate_img = DefaultDilateImage(
            img,
            threshold_px    =   threshold_px,
            kernel          =   kernel,
            kernel_area     =   kernel_area,
            is_binary_inv   =   is_binary_inv,
            is_otsu         =   is_otsu,
            )
    contour = GetContours(dilate_img)
    contour = SortContours_Width(contour=contour,is_reverse=is_reverse)
    return contour

def GetDefaultContours_Height(
        img,
        threshold_px    =   None,
        kernel          =   np.ones((2,30)),
        kernel_area     =   9,
        is_binary_inv   =   True,
        is_otsu         =   True,
        is_reverse=False):
    dilate_img = DefaultDilateImage(
            img,
            threshold_px    =   threshold_px,
            kernel          =   kernel,
            kernel_area     =   kernel_area,
            is_binary_inv   =   is_binary_inv,
            is_otsu         =   is_otsu,
            )
    contour = GetContours(dilate_img)
    contour = SortContours_Height(contour=contour,is_reverse=is_reverse)
    return contour

def GetDefaultContours_Area(
        img,
        threshold_px    =   None,
        kernel          =   np.ones((2,30)),
        kernel_area     =   9,
        is_binary_inv   =   True,
        is_otsu         =   True,
        is_reverse=False):
    dilate_img = DefaultDilateImage(
            img,
            threshold_px    =   threshold_px,
            kernel          =   kernel,
            kernel_area     =   kernel_area,
            is_binary_inv   =   is_binary_inv,
            is_otsu         =   is_otsu,
            )
    contour = GetContours(dilate_img)
    contour = SortContours_Area(contour=contour,is_reverse=is_reverse)
    return contour 

#####################################################################################################################

def DrawContours(img,contour,rgb=(255,0,0),is_show=False):
    img = ColorImage(img)
    img = cv2.drawContours(img, contour, -1, rgb, 3)
    if is_show==True:
        show.ShowImage(img,'DrawContours')
    return img 

def DrawDilateContours(dilate_img,img=None,rgb=(255,0,0),is_show=False):
    contours = GetContours(dilate_img)
    if not isinstance(img,np.ndarray):
        output_img = DrawContours(img=dilate_img,contour=contours,rgb=rgb,is_show=is_show)
        return output_img
    else:
        output_img = DrawContours(img=img,contour=contours,rgb=rgb,is_show=is_show)
        return output_img

def DrawDefaultContours(img,rgb=(255,0,0),threshold_px=None,is_binary_inv=True,kernel = np.ones((2,30)),is_original_img=True,is_otsu=True,is_show=False):
    dilate_img = DefaultDilateImage(
        img,
        threshold_px=threshold_px,
        is_binary_inv=is_binary_inv,
        kernel = kernel,
        is_otsu = is_otsu)
    if is_original_img==True:
        output_img = DrawDilateContours(dilate_img=dilate_img,img=img,rgb=rgb,is_show=is_show)
        return output_img
    else:
        output_img = DrawDilateContours(dilate_img=dilate_img,img=None,rgb=rgb,is_show=is_show)
        return output_img

#####################################################################################################################
