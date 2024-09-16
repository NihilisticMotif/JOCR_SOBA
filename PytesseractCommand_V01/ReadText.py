import pytesseract
import os
import cv2
from TextUtility import Path2Image

def GetImgText(img):
    # https://youtu.be/4uWp6dS6_G4?si=puVTzBm6vtLt4LXl
    img = Path2Image(img)
    return pytesseract.image_to_string(img)

def PrintImgText(img):
    text = GetImgText(img)
    print(text)

def SaveText(text,title='TextResult',folder='TextResult',fileformat='txt',is_show=False):
    # https://www.w3schools.com/python/python_file_write.asp
    # https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists/
    if not os.path.exists(folder):
        os.makedirs(folder)
    if fileformat[0]=='.':
        fileformat = fileformat[1:]
    path = os.path.join(folder,title+'.'+fileformat)
    file = open(path, 'w')
    file.write(text)
    file.close()
    if is_show==True:
        print(text)

def SaveImgText(img,title='textresult',folder='TextResult',fileformat='txt',is_show=False):
    # https://www.w3schools.com/python/python_file_write.asp
    # https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists/
    text=GetImgText(img)
    SaveText(text,title=title,folder=folder,fileformat=fileformat,is_show=is_show)



