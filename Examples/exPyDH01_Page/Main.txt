import sys
PytesseractPath = '/Users/imac/Desktop/JOCR_SOBA/PytesseractCommand'
sys.path.insert(1,PytesseractPath)

import ReadText as read

img_paths=[
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/OriginalImage/img.jpg' ,
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/Image/OtsuBinaryPx.jpg',
    '/Users/imac/Desktop/JOCR_SOBA/exPyDH01_Page/Image/ThickFont.jpg',
]

read.SaveTextFromImage(img_paths[0],'Original')
read.SaveTextFromImage(img_paths[1],'OtsuBinaryPx')
read.SaveTextFromImage(img_paths[2],'ThickFont')

'''
python3 Main.py
'''