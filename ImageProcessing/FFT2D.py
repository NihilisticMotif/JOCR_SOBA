import numpy as np
import cv2
import os
import GrayImage
import math
import Kernel2D

class FFT2D:
    def __init__(self,
            img:np.ndarray,
            name:str        =   'FFT', 
            folder:str      =   'FFT', 
            fileformat:str  =   'jpg'):
        self.dft = self.__GetFFT(img)
        self.name = name
        self.folder = folder           
        self.fileformat = fileformat   
    
    def __GetFFT(self,img):
        # https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
        img = img / np.float32(255)
        # dft is the array that contains complex numbers.
        dft = np.fft.fft2(img)          
        # zero frequency component of dft will be at top left corner. 
        # If you want to bring it to center, you need to shift the result n/2 in both dorection 
        # using np.fft.fftshift(). 
        dft = np.fft.fftshift(dft)  
        # By default, the transform is computed over the last two axes of the input array
        # Implies that in default case, the shape of dft is the same as shape of img.
        return dft

    def __Draw(self):
        dft = np.abs(self.dft)
        dft = dft/(255.0**2)
        dft = dft ** (1/4)
        return dft
    
    def __CheckValidShape(self,
                update_row:int,
                update_col:int):
        row = math.floor(self.dft.shape[0]/2)
        col = math.floor(self.dft.shape[1]/2)
        if update_row > row or update_col > col:
            print('kernel size is invalid')
            print('shape of dft',self.dft.shape)
            print('center_row',row)
            print('center_col',col)
            print('update_row',update_row)
            print('update_col',update_col)
            return False 
        return True
    
    def Show(self):
        cv2.imshow(self.name,self.__Draw())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Save(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        if self.fileformat[0]=='.':
            self.fileformat = self.fileformat[1:]
        img_path = os.path.join(
            self.folder,self.name+'.'+self.fileformat)
        self.__Draw().save(img_path)
    
    def GetIFFT(self):
        dft = np.fft.ifftshift(self.dft)
        img = np.fft.ifft2(dft)
        img = np.real(img)
        return GrayImage(img,self.name,self.folder,self.fileformat)

    def Edit(self,
            update_row:int,
            update_col:None|int,
            new_frequency:float = 0,
            type = cv2.MORPH_RECT,
            is_sharp:bool|None=True):
        row = math.floor(self.dft.shape[0]/2)
        col = math.floor(self.dft.shape[1]/2)
        if self.__CheckValidShape(update_row,update_col):
            kernel = Kernel2D(
                width=update_row,
                height=update_col,
                type=type)
            kernel = kernel.GetKernel()
            mask = np.zeros(self.dft.shape)
            mask[   row-update_row:row+update_row, 
                    col-update_col:col+update_col   ] = kernel.T
            if is_sharp == True:
                mask = np.where(mask < 1,1,new_frequency)
            if is_sharp == False:
                mask = np.where(mask < 1,new_frequency,1)
            self.dft *= mask
  
    def KernelEdit(self,
            kernel:np.ndarray,
            is_sharp:bool|None=None,
            new_frequency:float=0):
        row = math.floor(self.dft.shape[0]/2)
        col = math.floor(self.dft.shape[1]/2)
        if self.__CheckValidShape(kernel.shape[0],kernel.shape[1]):
            update_row = kernel.shape[0]
            update_col = kernel.shape[0]
            mask[   row-update_row:row+update_row, 
                    col-update_col:col+update_col   ] = kernel.T
            if is_sharp == True:
                mask = np.where(mask < 1,1,new_frequency)
            if is_sharp == False:
                mask = np.where(mask < 1,new_frequency,1)
            self.dft *= mask

    def GetIFFT(self):
        dft = np.fft.ifftshift(self.dft)
        img = np.fft.ifft2(dft)
        img = np.real(img)
        img = ( 255 * img ).astype(np.uint8) 
        return img
    
    def Help():
        message ='''
Note that
* Time - Amplitude               Frequency - Amplitude         
* amplitude fast in short time = high frequency signal
* amplitude slow in long  time = low  frequency signal
* Desmos Simulation: https://www.desmos.com/calculator/utr2n8y1ja
* We apply this idea for 2D image processing task.

Sharp FFT

[[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]

Blur FFT

[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

Other Learning Resource
1. The Discrete Fourier Transform: Most Important Algorithm Ever?
* https://youtu.be/yYEMxqreA10?si=-MZ6QTnQq0DmxV06
2. he Remarkable Story Behind The Most Important Algorithm Of All Time
* https://youtu.be/nmgFG7PUHfo?si=WYsNGYudyPnMzE4c
3. The Fourier Series and Fourier Transform Demystified
* https://youtu.be/mgXSevZmjPc?si=X-V8HRe0QYVm8QUL

By: ImageProcessing / FFT2D.py
'''
        print(message)