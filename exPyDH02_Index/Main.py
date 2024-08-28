import sys
PytesseractPath = '/Users/imac/Desktop/JOCR_SOBA/PytesseractCommand'
sys.path.insert(1,PytesseractPath)

import ReadText as read 
from ReadColumn import SaveColumnText,SaveMultiColumnText

img_paths='/Users/imac/Desktop/JOCR_SOBA/exPyDH02_Index/OriginalImage/img.jpeg'
ocr_output = SaveMultiColumnText(
    img_paths,
    width_and_height=[
        [20,200]
        ],
    is_show=True,
    title='TextFromImgWithColumns01'
    )
ocr_output = str(ocr_output)
read.SaveText(ocr_output,'textresult02')

'''
python3 Main.py
'''