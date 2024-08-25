import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)

from ShowImage import ReadImage
import cv2

def Path2Image(img):
    if type(img)==str:
        return ReadImage(img)
    return img
