import cv2
import pytesseract
from pytesseract import Output
import preprocessing as process_img

################################################
#   Preprocessing Image
################################################

path='/Users/imac/Desktop/JOCR_Dytecture/Image/JojoSoba/IMG_7563.jpeg'
img = cv2.imread(path)
img=process_img.get_grayscale(img)
img=process_img.WhiteBackGround(img)

################################################
#   Analyse Text
################################################

custom_config='l- thai+eng'
output = pytesseract.image_to_string(
    img, 
    config=custom_config
    )

################################################
#   Display Output
################################################

process_img.ShowMustGoOn(img)

print(output)


'''
python3 main.py
'''
