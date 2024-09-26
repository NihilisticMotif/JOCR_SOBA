import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)

from Contour import DetectContourImg
import numpy as np
from GrayImage import GrayImage
from Image import Image


class TableImage:
    def __init__(
            self,
            img:np.ndarray,
            threshold_px:None|int = None,
            kernel      :np.ndarray = np.ones((2,30)),
            kernel_area :int = 9
        ):
        self.img = Image(img)
        self.dilate_img = DetectContourImg(self.img, threshold_px, kernel, kernel_area)

