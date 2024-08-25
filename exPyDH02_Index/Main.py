import sys
PytesseractPath = '/Users/imac/Desktop/JOCR_SOBA/PytesseractCommand'
sys.path.insert(1,PytesseractPath)

import ReadText as read 
from Column import GetTextFromColumn

img_paths='/Users/imac/Desktop/JOCR_SOBA/exPyDH02_Index/OriginalImage/img.jpeg'
ocr_output = GetTextFromColumn(
    img_paths,
    width=20,
    height=200,
    is_a_column=False,
    is_show=True,
    save_image_folder_and_name='ImageWithColumn',
    is_save_multiple_imgs=True
    )
ocr_output = str(ocr_output)
read.SaveText(ocr_output,'textresult02')

'''
python3 Main.py
'''