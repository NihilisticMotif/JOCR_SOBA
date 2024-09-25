import numpy as np 
import math 
from Kernel2D import Kernel2D
from ImageUtility import GetSize
import cv2

'''
Note that
* Time - Amplitude               Frequency - Amplitude         
* amplitude fast in short time = high frequency signal
* amplitude slow in long  time = low  frequency signal
* Desmos Simulation: https://www.desmos.com/calculator/utr2n8y1ja
* We apply this idea for 2D image processing task.

Keyword
* freq = frequency
* cx = center x
* cy = center y

Other Learning Resource
1. The Discrete Fourier Transform: Most Important Algorithm Ever?
* https://youtu.be/yYEMxqreA10?si=-MZ6QTnQq0DmxV06
2. he Remarkable Story Behind The Most Important Algorithm Of All Time
* https://youtu.be/nmgFG7PUHfo?si=WYsNGYudyPnMzE4c
3. The Fourier Series and Fourier Transform Demystified
* https://youtu.be/mgXSevZmjPc?si=X-V8HRe0QYVm8QUL
'''

########################################################################################################################################################

def GetFFT(img:np.ndarray):
    scale=255
    # https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    img = img / np.float32(scale)
    # dft is the array that contains complex numbers.
    dft = np.fft.fft2(img,None) # fft_size)          
    # zero frequency component of dft will be at top left corner. 
    # If you want to bring it to center, you need to shift the result n/2 in both dorection 
    # using np.fft.fftshift(). 
    dft = np.fft.fftshift(dft)  
    # By default, the transform is computed over the last two axes of the input array
    # Implies that in default case, the shape of dft is the same as shape of img.
    return dft

def GetIFFT(dft:np.ndarray):
    dft = np.fft.ifftshift(dft)
    img = np.fft.ifft2(dft)
    img = np.real(img)
    return img

def GetFFTImage(img:np.ndarray):
    dft = GetFFT(img)
    dft = np.abs(dft)
    dft = dft/(255.0**2)
    dft = dft ** (1/4)
    return dft

########################################################################################################################################################

def PrivateEditFFT(dft:np.ndarray, row:int = None, col:int|None = None, is_blur:bool = True, freq:float = 0, mode:int = cv2.MORPH_RECT):
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # https://stackoverflow.com/questions/56594598/change-1s-to-0-and-0s-to-1-in-numpy-array-without-looping
    cx = math.floor(dft.shape[1]/2)
    cy = math.floor(dft.shape[0]/2)
    row = GetSize(row,cx,0)
    col = GetSize(col,cy,0)
    mask = np.zeros(dft.shape)
    kernel = Kernel2D(row*2, col*2, mode = mode)
    mask[cx-row:cx+row, 
         cy-col:cy+col] = kernel.T
    if is_blur == True:
        mask = np.where(mask < 1,freq,1)
    else:
        mask = np.where(mask < 1,1,freq)
    dft *= mask
    return dft 

def EditFFT(img:np.ndarray, row:int, col:int, freq:float = 0, mode:int = cv2.MORPH_RECT):
    dft = GetFFT(img)
    dft = PrivateEditFFT(dft,row,col,freq, mode)
    img = GetIFFT(dft)
    img = ( 255 * img ).astype(np.uint8) 
    return img

def FFTBlur(img:np.ndarray, row:int, col:int, freq:float = 0, mode:int = cv2.MORPH_RECT):
    return EditFFT(img, row, col, freq, mode, is_blur = True)

def FFTSharp(img:np.ndarray, row:int, col:int, freq:float = 0, mode:int = cv2.MORPH_RECT):
    return EditFFT(img, row, col, freq, mode, is_blur = False)