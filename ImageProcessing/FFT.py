from ShowImage import ShowImage 
from GrayImage import GrayImage
import cv2 as cv
import numpy as np
import math

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
    print('Extra Hands/Legs (Polymelia) : 1 in 1700')
    print('WARNING: updated_rows and/or updated_cols is invalid.')
    print('dft.shape',dft.shape)
    print('center_rows',center_rows)
    print('updated_rows',updated_rows)
    print('center_cols',center_cols)
    print('updated_cols',updated_cols)
    print('Reported by ImageProcessing / FFT.py / def '+function_name)
    print('Reported by ImageProcessing / FFT.py / def FFTExtraHandsWarning(dft,updated_rows,updated_cols,function_name)')

def PrepareSpectrum(spectrum):
    # https://stackoverflow.com/questions/65369482/issue-with-displaying-dft-output-with-opencvs-imshow
    mag = np.abs(spectrum)
    mag /= mag.max()
    mag = mag ** (1/4)
    return mag

#def MaximumFequency(img):

def ReturnFFT(img,is_show=False,fft_size=None):
    # https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    img = GrayImage(img)
    img = img / np.float32(img.max())
    # dft is the array that contains complex numbers.
    dft = np.fft.fft2(img,fft_size)          
    dft = PrepareSpectrum(dft)
    # zero frequency component of dft will be at top left corner. 
    # If you want to bring it to center, you need to shift the result n/2 in both dorection 
    # using np.fft.fftshift(). 
    dft = np.fft.fftshift(dft)  
    if is_show==True:
        # By default, the transform is computed over the last two axes of the input array
        # Implies that in default case, the shape of dft is the same as shape of img.
        ShowImage(dft)
    return dft

def ReturnIFFT(dft,max_px=None,is_show=False):
    if is_show==True:
        ShowImage(dft)
    dft = np.fft.ifftshift(dft)
    img = np.fft.ifft2(dft)
    img = np.real(img)
    return img

########################################################################################################################################################

def FFTRectangleSharpen(dft,updated_rows,updated_cols,new_frequency=0):
    center_rows = math.floor(dft.shape[0]/2)
    center_cols = math.floor(dft.shape[1]/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTRectangleSharpen(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    dft[center_rows-updated_rows:center_rows+updated_rows, 
        center_cols-updated_cols:center_cols+updated_cols] = new_frequency
    return dft

def FFTSharpen01(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    if is_show==True:
        print('FFTSharpen01 is based on FFTRectangleSharpen.')
    dft = ReturnFFT(img,is_show=is_show)
    dft = FFTRectangleSharpen(dft,updated_rows,updated_cols,new_frequency)
    img = ReturnIFFT(dft,is_show=is_show)
    return img

########################################################################################################################################################

def FFTRectangleBlur(dft,updated_rows,updated_cols,new_frequency=0):
    rows = dft.shape[0]
    cols = dft.shape[1]    
    center_rows = math.floor(dft.shape[0]/2)
    center_cols = math.floor(dft.shape[1]/2)
    if updated_rows>=center_rows or updated_cols>=center_cols:
        warning='FFTRectangleBlur(dft,updated_rows,updated_cols,new_frequency=0)'
        FFTExtraHandsWarning(dft,updated_rows,updated_cols,warning)
        return dft
    dft[0:center_rows-updated_rows,
        0:cols]=new_frequency
    dft[center_rows+updated_rows:rows,
        0:cols]=new_frequency
    dft[0:-rows,
        0:center_cols-updated_cols]=new_frequency
    dft[0:rows,
        center_cols+updated_cols:cols]=new_frequency
    return dft

def FFTBlur01(img,updated_rows,updated_cols,new_frequency=0,is_show=False):
    if is_show==True:
        print('FFTBlur01 is based on FFTRectangleBlur.')
    dft = ReturnFFT(img,is_show=is_show)
    dft = FFTRectangleBlur(dft,updated_rows,updated_cols,new_frequency)
    img = ReturnIFFT(dft,is_show=is_show)
    img = 255 * img
    #img = 255 * img
    return img

########################################################################################################################################################

def FFTCircleSharpen():
    pass

def FFTSharpen02():
    pass

########################################################################################################################################################

def FFTCircleBlur():
    pass

def FFTBlur02():
    pass
