import sys
ImageProcessingPath = '/Users/imac/Desktop/JOCR_SOBA/ImageProcessing'
sys.path.insert(1, ImageProcessingPath)

import Contour as tour
import ReadText as read
import cv2
import numpy as np
from TextUtility import Path2Image
from GrayImage import ColorImage
import ShowImage as show

def GetTextFromColumn(
        img,
        width,
        height,
        dilate_img      =   None,
        is_a_column     =   True,
        display_color   =   [
            (255,0,0),
            (0,255,0),
            (0,0,255),
        ],
        threshold_px    =   None,
        kernel          =   np.ones((13,3)),
        kernel_area     =   9,
        is_binary_inv   =   True,
        is_otsu         =   True,
        is_show         =   False,
        save_image_folder_and_name =   None,
        is_save_multiple_imgs      =   False
    ):

    #####################################################################################################################

    img = Path2Image(img)
    if not isinstance(dilate_img,np.ndarray):
        dilate_img = tour.DefaultDilateImage(
            img,
            threshold_px    =   threshold_px,
            kernel          =   kernel,
            kernel_area     =   kernel_area,
            is_binary_inv   =   is_binary_inv,
            is_otsu         =   is_otsu,
            )
    contours = tour.GetContours_Left2Right(dilate_img,False)

    #####################################################################################################################

    text_result = []
    count_columns = 0
    sub_imgs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w>width and h>height:
            sub_img=img[y:y+h, x:x+w]
            if is_save_multiple_imgs==True:
                sub_imgs.append(img[y:y+h, x:x+w])
            output = read.GetText(sub_img)
            output = output.split('\n')
            show_img = cv2.rectangle(img, (x, y), (x+w, y+h), display_color[count_columns%len(display_color)], 2)
            if is_show==True:
                img = ColorImage(img)
                show.ShowImage(show_img,'Full Image with Columns.')
                show.ShowImage(sub_img,'Image with just 1 Column.')
                count_columns+=1
            if is_a_column==True:
                text_result.extend(output)
            else:
                text_result.append(output)
    
    #####################################################################################################################

    if is_show==True:
        print('Number of all columns:',count_columns)
    if isinstance(save_image_folder_and_name,str):
        show.SaveImage(show_img,save_image_folder_and_name)
        if is_save_multiple_imgs==True:
            ii=0
            for sub_img in sub_imgs:
                show.SaveImage(sub_img,save_image_folder_and_name+'No'+str(ii))
                ii+=1
    if isinstance(save_image_folder_and_name,list):
        print('type(save_image_folder_and_name)',save_image_folder_and_name)
        if all(isinstance(item, str) for item in save_image_folder_and_name):
            # https://stackoverflow.com/questions/37357798/how-to-check-if-all-items-in-list-are-string
            if len(save_image_folder_and_name)==1:
                show.SaveImage(show_img,save_image_folder_and_name[0])
                if is_save_multiple_imgs==True:
                    ii=0
                    for sub_img in sub_imgs:
                        show.SaveImage(
                            sub_img,                    
                            save_image_folder_and_name[0]+'No'+str(ii),
                        )
                        ii+=1
            if len(save_image_folder_and_name)==2:
                show.SaveImage(
                    show_img,
                    save_image_folder_and_name[0],
                    folder=save_image_folder_and_name[1]
                    )
                if is_save_multiple_imgs==True:
                    ii=0
                    for sub_img in sub_imgs:
                        show.SaveImage(
                            sub_img,                    
                            save_image_folder_and_name[0]+'No'+str(ii),
                            folder=save_image_folder_and_name[1],
                        )
                        ii+=1
            if len(save_image_folder_and_name)>2:
                show.SaveImage(
                    show_img,
                    save_image_folder_and_name[0],
                    folder=save_image_folder_and_name[1],
                    fileformat=save_image_folder_and_name[2],
                    )
                if is_save_multiple_imgs==True:
                    ii=0
                    for sub_img in sub_imgs:
                        show.SaveImage(
                            sub_img,                    
                            save_image_folder_and_name[0]+'No'+str(ii),
                            folder=save_image_folder_and_name[1],
                            fileformat=save_image_folder_and_name[2],
                        )
                        ii+=1
    return text_result

'''
python3 Column.py
'''