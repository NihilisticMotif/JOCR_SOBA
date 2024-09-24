import numpy as np 
import math 
from Kernel2D import Kernel2D
import cv2

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

def GetFFT(img,fft_size=None,scale=255):
    # https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    img = img / np.float32(scale)
    # dft is the array that contains complex numbers.
    dft = np.fft.fft2(img,fft_size)          
    # zero frequency component of dft will be at top left corner. 
    # If you want to bring it to center, you need to shift the result n/2 in both dorection 
    # using np.fft.fftshift(). 
    dft = np.fft.fftshift(dft)  
    # By default, the transform is computed over the last two axes of the input array
    # Implies that in default case, the shape of dft is the same as shape of img.
    return dft

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

def GetIFFT(dft):
    dft = np.fft.ifftshift(dft)
    img = np.fft.ifft2(dft)
    img = np.real(img)
    return img

def PrivateEditFFT(dft,updated_rows,updated_cols, is_blur = True, new_frequency=0, mode = cv2.MORPH_RECT):
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
    kernel = Kernel2D(updated_rows*2,updated_cols*2,mode = mode)
    mask[center_rows-updated_rows:center_rows+updated_rows, 
    center_cols-updated_cols:center_cols+updated_cols] = kernel.T
    if is_blur == True:
        mask = np.where(mask < 1,new_frequency,1)
    else:
        mask = np.where(mask < 1,1,new_frequency)
    dft *= mask
    return dft 

def EditFFT(img, updated_rows, updated_cols, new_frequency = 0, mode = cv2.MORPH_RECT):
    dft = GetFFT(img)
    dft = PrivateEditFFT(dft,updated_rows,updated_cols,new_frequency, mode)
    img = GetIFFT(dft)
    img = ( 255 * img ).astype(np.uint8) 
    return img

def FFTBlur(img, updated_rows, updated_cols, new_frequency = 0, mode = cv2.MORPH_RECT):
    return EditFFT(img, updated_rows, updated_cols, new_frequency, mode, is_blur = True)

def FFTSharp(img, updated_rows, updated_cols, new_frequency = 0, mode = cv2.MORPH_RECT):
    return EditFFT(img, updated_rows, updated_cols, new_frequency, mode, is_blur = False)