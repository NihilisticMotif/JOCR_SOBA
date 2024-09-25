import cv2

def GetContours(dilate_img):
    contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def SortContours_Left2Right(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[0], reverse = is_reverse)

def SortContours_Top2Bottom(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[1], reverse = is_reverse)

def SortContours_Width(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[2], reverse = is_reverse)

def SortContours_Height(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.boundingRect(x)[3], reverse = is_reverse)

def SortContours_Area(contour:list, is_reverse:bool = False):
    return sorted(contour, key=lambda x:cv2.contourArea(x), reverse = is_reverse)

