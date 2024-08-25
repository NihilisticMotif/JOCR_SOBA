from ShowImage import ShowImage, SaveImage
from GrayImage import GrayImage
import cv2
import numpy as np
import math
from ImageUtility import Numpy2Image
import os

########################################################################################################################################################

'''
Note that
* Time - Amplitude               Frequency - Amplitude         
* amplitude fast in short time = high frequency signal
* amplitude slow in long  time = low  frequency signal
* Desmos Simulation: https://www.desmos.com/calculator/utr2n8y1ja
* We apply this idea for 2D image processing task.

Other Learning Resource
1. The Discrete Fourier Transform: Most Important Algorithm Ever?
* https://youtu.be/yYEMxqreA10?si=-MZ6QTnQq0DmxV06
2. he Remarkable Story Behind The Most Important Algorithm Of All Time
* https://youtu.be/nmgFG7PUHfo?si=WYsNGYudyPnMzE4c
3. The Fourier Series and Fourier Transform Demystified
* https://youtu.be/mgXSevZmjPc?si=X-V8HRe0QYVm8QUL
'''

########################################################################################################################################################

def FFTExtraHandsWarning(dft,updated_rows,updated_cols,function_name):
    center_rows = math.floor(dft.shape[0]/2)
    center_cols = math.floor(dft.shape[1]/2)
    # https://youtu.be/mI9FIugGIZQ?si=KnaCetRmaAbosYeT
    print()
    print('Extra Hands/Legs (Polymelia) : 1 in 1700')
    print('WARNING: updated_rows and/or updated_cols is invalid.')
    print('dft.shape____:',dft.shape)
    print('center_rows__:',center_rows)
    print('updated_rows_:',updated_rows)
    print('center_cols__:',center_cols)
    print('updated_cols_:',updated_cols)
    print('Reported by ImageProcessing / FFT.py / def '+function_name)
    print('Reported by ImageProcessing / FFT.py / def FFTExtraHandsWarning')
    print()

def AdjustFFTForDisplay(dft):
    dft = np.abs(dft)
    dft = dft/(255.0**2)
    dft = dft ** (1/4)
    return dft

def ShowFFT(dft,is_show):
    # https://stackoverflow.com/questions/65369482/issue-with-displaying-dft-output-with-opencvs-imshow
    if is_show==True:
        dft = AdjustFFTForDisplay(dft)
        ShowImage(dft)

def SaveFFT(img,img_title,folder='FFT',fileformat='jpg',is_editfft=False):
    if is_editfft==False:
        img = GetFFT(img)
    img = AdjustFFTForDisplay(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img*255).astype(np.uint8) 
    SaveImage(img,img_title,folder=folder,fileformat=fileformat)

def GetFFT(img,is_show=False,fft_size=None):
    # https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    img = GrayImage(img)
    img = img / np.float32(255)
    # dft is the array that contains complex numbers.
    dft = np.fft.fft2(img,fft_size)          
    # zero frequency component of dft will be at top left corner. 
    # If you want to bring it to center, you need to shift the result n/2 in both dorection 
    # using np.fft.fftshift(). 
    dft = np.fft.fftshift(dft)  
    # By default, the transform is computed over the last two axes of the input array
    # Implies that in default case, the shape of dft is the same as shape of img.
    ShowFFT(dft,is_show=is_show)
    return dft

def GetIFFT(dft,is_show=False):
    ShowFFT(dft,is_show=is_show)
    dft = np.fft.ifftshift(dft)
    img = np.fft.ifft2(dft)
    img = np.real(img)
    return img

########################################################################################################################################################

'''
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
'''

def FFTRectangleSharpen(dft,updated_rows,updated_cols,new_frequency=0):
    rows = dft.shape[0]
    cols = dft.shape[1]    
    center_rows = math.floor(rows/2)
    center_cols = math.floor(cols/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTRectangleSharpen(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    mask = np.ones((rows,cols), dtype=np.uint8)
    mask[center_rows-updated_rows:center_rows+updated_rows, 
        center_cols-updated_cols:center_cols+updated_cols] *= new_frequency
    dft *= mask
    return dft

def FFTSharpen01(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    # https://stackoverflow.com/questions/16720682/pil-cannot-write-mode-f-to-jpeg
    if is_show==True:
        print('FFTSharpen01 is based on FFTRectangleSharpen.')
    dft = GetFFT(img,is_show=is_show)
    dft = FFTRectangleSharpen(dft,updated_rows,updated_cols,new_frequency)
    img = GetIFFT(dft,is_show=is_show)
    img = ( 255 * img ).astype(np.uint8) 
    return img

########################################################################################################################################################

'''
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
'''

def FFTRectangleBlur(dft,updated_rows,updated_cols,new_frequency=0):
    rows = dft.shape[0]
    cols = dft.shape[1]    
    center_rows = math.floor(rows/2)
    center_cols = math.floor(cols/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTRectangleBlur(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    mask = np.ones((rows,cols), dtype=np.uint8)*new_frequency
    mask[center_rows-updated_rows:center_rows+updated_rows, 
         center_cols-updated_cols:center_cols+updated_cols] = 1
    dft *= mask
    return dft

def FFTBlur01(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    if is_show==True:
        print('FFTBlur01 is based on FFTRectangleBlur.')
    dft = GetFFT(img,is_show=is_show)
    dft = FFTRectangleBlur(dft,updated_rows,updated_cols,new_frequency)
    img = GetIFFT(dft,is_show=is_show)
    img = ( 255 * img ).astype(np.uint8) 
    return img

########################################################################################################################################################

'''
[[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 0 0 0 0 0 1 1 1 1 1]
 [1 1 1 0 0 0 0 0 0 0 0 0 1 1 1]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 1 0 0 0 0 0 0 0 0 0 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]
'''

def FFTCircleSharpen(dft,updated_rows,updated_cols,new_frequency=0):
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # https://stackoverflow.com/questions/56594598/change-1s-to-0-and-0s-to-1-in-numpy-array-without-looping
    rows = dft.shape[0]
    cols = dft.shape[1]    
    center_rows = math.floor(rows/2)
    center_cols = math.floor(cols/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTCircleSharpen(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    mask = np.zeros(dft.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (updated_rows*2, updated_cols*2))
    mask[center_rows-updated_rows:center_rows+updated_rows, 
    center_cols-updated_cols:center_cols+updated_cols] = kernel.T
    mask = np.where(mask < 1,1,new_frequency)
    dft *= mask
    return dft 

def FFTSharpen02(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    if is_show==True:
        print('FFTBlur01 is based on FFTCircleSharpen.')
    dft = GetFFT(img,is_show=is_show)
    dft = FFTCircleSharpen(dft,updated_rows,updated_cols,new_frequency)
    img = GetIFFT(dft,is_show=is_show)
    img = ( 255 * img ).astype(np.uint8) 
    return img

########################################################################################################################################################

'''
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 1 1 1 1 0 0 0 0 0]
 [0 0 0 1 1 1 1 1 1 1 1 1 0 0 0]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 0 1 1 1 1 1 1 1 1 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
'''

def FFTCircleBlur(dft,updated_rows,updated_cols,new_frequency=0):
    rows = dft.shape[0]
    cols = dft.shape[1]    
    center_rows = math.floor(rows/2)
    center_cols = math.floor(cols/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTCircleBlur(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    mask = np.zeros(dft.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (updated_rows*2, updated_cols*2))
    mask[center_rows-updated_rows:center_rows+updated_rows, 
    center_cols-updated_cols:center_cols+updated_cols] = kernel.T
    mask = np.where(mask < 1,new_frequency,1)
    dft *= mask
    return dft 

def FFTBlur02(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    if is_show==True:
        print('FFTBlur01 is based on FFTCircleBlur.')
    dft = GetFFT(img,is_show=is_show)
    dft = FFTCircleBlur(dft,updated_rows,updated_cols,new_frequency)
    img = GetIFFT(dft,is_show=is_show)
    img = ( 255 * img ).astype(np.uint8) 
    return img

########################################################################################################################################################

'''
[[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]
'''

def FFTCrossSharpen(dft,updated_rows,updated_cols,new_frequency=0):
    rows = dft.shape[0]
    cols = dft.shape[1]    
    center_rows = math.floor(rows/2)
    center_cols = math.floor(cols/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTCrossSharpen(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    mask = np.zeros(dft.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (updated_rows*2, updated_cols*2))
    mask[center_rows-updated_rows:center_rows+updated_rows, 
    center_cols-updated_cols:center_cols+updated_cols] = kernel.T
    mask = np.where(mask < 1,1,new_frequency)
    dft *= mask
    return dft 

def FFTSharpen03(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    if is_show==True:
        print('FFTBlur01 is based on FFTCrossSharpen.')
    dft = GetFFT(img,is_show=is_show)
    dft = FFTCrossSharpen(dft,updated_rows,updated_cols,new_frequency)
    img = GetIFFT(dft,is_show=is_show)
    img = ( 255 * img ).astype(np.uint8) 
    return img

########################################################################################################################################################

'''
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
'''
def FFTCrossBlur(dft,updated_rows,updated_cols,new_frequency=0):
    rows = dft.shape[0]
    cols = dft.shape[1]    
    center_rows = math.floor(rows/2)
    center_cols = math.floor(cols/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTCrossBlur(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    mask = np.zeros(dft.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (updated_rows*2, updated_cols*2))
    mask[center_rows-updated_rows:center_rows+updated_rows, 
    center_cols-updated_cols:center_cols+updated_cols] = kernel.T
    mask = np.where(mask < 1,new_frequency,1)
    dft *= mask
    return dft 

def FFTBlur03(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    if is_show==True:
        print('FFTBlur01 is based on FFTCrossBlur.')
    dft = GetFFT(img,is_show=is_show)
    dft = FFTCrossBlur(dft,updated_rows,updated_cols,new_frequency)
    img = GetIFFT(dft,is_show=is_show)
    img = ( 255 * img ).astype(np.uint8) 
    return img

########################################################################################################################################################

