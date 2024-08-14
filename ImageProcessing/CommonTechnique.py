import cv2
import numpy as np

img = cv2.imread('image.jpg')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image,n=5):
    return cv2.medianBlur(image,n)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image,kernel=None):
    if kernel==None:
        kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def Sharpen(img):
    k2=-0.1
    k1=-5
    k0=-(k2*16+k1*8)+1
    kernel = np.array([
        [k2,k2,k2,k2,k2,], 
        [k2,k1,k1,k1,k2,], 
        [k2,k1,k0,k1,k2,], 
        [k2,k1,k1,k1,k2,], 
        [k2,k2,k2,k2,k2,]
        ])
    # https://youtu.be/KuXjwB4LzSA?si=mt-leKGKjpMnJGfg
    # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    return cv2.filter2D(img, -1, kernel)

def WhiteBackGround(img):
    RangeMax=100
    A1=1
    Step=1
    Start=0
    for i in range(RangeMax):
      mark=np.logical_and(img>Start+(i*Step)/A1,img<Start+((i+1)*Step)/A1)
      img[mark]=Start+(i*Step)/A1
    img[img>=Start+((RangeMax)*Step)/A1]=255
    return img

def ShowImage(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ImportTest():
    print("This is 80000Hours Podcast.")

# https://nanonets.com/blog/ocr-with-tesseract/