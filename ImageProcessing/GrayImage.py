import Image 
import numpy as np 
import cv2
from ImageFunction import OddKernelArea
import Threshold
import FFT2D
class GrayImage(Image):
    def __init__(self, 
            img:np.ndarray | str, 
            name:str        =   'Image', 
            folder:str      =   'Image', 
            fileformat:str  =   'jpg'):
        super().__init__(self, img, name, folder, fileformat)
        if type(img) == str:
            self.img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_RGB2GRAY)
        elif len(img.shape)==3:
            self.img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape)==2:
            self.img = img

    def Copy(self):
        output_image = GrayImage(
            self.img,
            self.name,
            self.folder,
            self.fileformat)
        return output_image 

    def CreateBorders(
            self,
            size  : int =50,
            color : int = 0):
        top, bottom, left, right = [size]*4
        self.img = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    def Color(self):
        return Image(self.img,self.name,self.folder,self.fileformat)

    def Inverted(self):
        self.img = cv2.bitwise_not(self.img)

    def RemoveNoise(
            self, 
            kernel: np.ndarray = np.ones((1, 1), np.uint8), 
            kernel_area:int = 3, 
            dilate_iter:int = 1, 
            erode_iter:int  = 1):
        kernel_area = OddKernelArea(kernel_area)
        # https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb
        # https://www.geeksforgeeks.org/erosion-and-dilation-morphological-transformations-in-opencv-in-cpp/
        self.img = cv2.dilate(self.img, kernel, iterations=dilate_iter)
        self.img = cv2.erode(self.img, kernel,  iterations=erode_iter )
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.img = cv2.medianBlur(self.img, kernel_area)

    def ThinFont(
            self, 
            kernel: np.ndarray = np.ones((2,2),np.uint8), 
            iter:int = 1):
        self.img = cv2.bitwise_not(self.img)
        self.img = cv2.erode(self.img, kernel, iterations=iter)
        self.img = cv2.bitwise_not(self.img)
        return self.img

    def ThickFont(
            self, 
            kernel: np.ndarray = np.ones((2,2),np.uint8), 
            iter:int = 1):
        self.img = cv2.bitwise_not(self.img)
        self.img = cv2.dilate(self.img, kernel, iterations=iter)
        self.img = cv2.bitwise_not(self.img)
        return self.img

    def Threshold(
            self,
            threshold: Threshold):
        if type(threshold.threshold_px) == int:
            self.img = cv2.threshold(
                self.img,
                threshold.threshold_px,
                threshold.max_px,
                threshold.method,
            )[1]
        if threshold.threshold_px == cv2.THRESH_OTSU:
            self.img = cv2.threshold(
                self.img,
                0,
                threshold.max_px,
                threshold.method+cv2.THRESH_OTSU)[1]
        if threshold.threshold_px in [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]:
            self.img = cv2.adaptiveThreshold(
                self.img,
                threshold.max_px,
                threshold.threshold_px,
                threshold.method,
                threshold.kernel_area,
                threshold.constant)
    
    def Filder2D(
            self,
            kernel:np.ndarray,
            scalar:float = 1):
        self.img = cv2.filter2D(self.img, -1, scalar * kernel)
    
    def MeanBlur(
            self,
            kernel_area:int=15,
            scalar:None|float = None
            ):
        # Update each pixel value to average pixel value with in the kernel_area to Blur image
        # https://youtu.be/KuXjwB4LzSA?si=mt-leKGKjpMnJGfg
        # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
        kernel_area=OddKernelArea(kernel_area)
        if scalar==None:
            scalar=(1/kernel_area)**2
        kernel = scalar*np.ones((kernel_area,kernel_area))
        self.img = cv2.filter2D(self.img, -1, kernel)

    def GaussBlur(self,kernel_area:int=15):
        # Blur the image based in the pixel with in kernel_area using Gaussian function
        # https://www.geeksforgeeks.org/python-image-blurring-using-opencv/
        kernel_area=OddKernelArea(kernel_area)
        self.img = cv2.GaussianBlur(self.img,(kernel_area,kernel_area), 0) 

    def BilateralBlur(self,kernel_area:int=15,effect:int=75):
        # Remove the noise and preserve the edge.
        # https://youtu.be/LjbYKWAQA5s?si=1br6Rl9OYkTZQPJB
        # https://www.tutorialspoint.com/opencv/opencv_bilateral_filter.htm
        kernel_area=OddKernelArea(kernel_area)
        self.img = cv2.bilateralFilter(self.img, kernel_area, effect, effect)

    def GetFFT(self):
        return FFT2D(
            self.img,
            self.name+'_FFT',
            self.folder,
            self.fileformat)
    
    def ShowFFT(self):
        dft = self.GetFFT()
        dft.Show()

    def SaveFFT(self):
        dft = self.GetFFT()
        dft.Save()
    
    def EditFFT(self,
            update_row:int,
            update_col:None|int,
            new_frequency:float = 0,
            type = cv2.MORPH_RECT,
            is_sharp:bool|None=True):
        dft = self.GetFFT()
        dft.Edit(
            update_row    = update_row    ,
            update_col    = update_col    ,
            new_frequency = new_frequency ,
            type          = type          ,
            is_sharp      = is_sharp      )
        self.img = dft.GetIFFT()

    def KernelEditFFT(self,
            kernel:np.ndarray,
            is_sharp:bool|None=None,
            new_frequency:float=0):
        dft = self.GetFFT()
        dft.KernelEdit(
            kernel          =   kernel,
            is_sharp        =   is_sharp,
            new_frequency   = new_frequency)
        self.img = dft.GetIFFT()
