import ShowImage as show 
import Contour as tour
import GrayImage as gray
import numpy as np 
import cv2

def ReturnMaxWidth(max_width,img):
    if not isinstance(max_width,(int,float)) or max_width > img.shape[1]:
        return img.shape[0]
    return max_width

def ReturnMaxHeight(max_height,img):
    if not isinstance(max_height,(int,float)) or max_height > img.shape[1]:
        return img.shape[1]
    return max_height

def ColumnSegmentation(
        img,
        width,
        height,
        max_width     =   None,
        max_height    =   None,
        dilate_img      =   None,
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
        img_title       =   None,
        folder          =   'ColumnSegment',
        fileformat      =   'jpg',
        is_multiple_imgs=   False,
        is_return_img_ls=   False
    ):

    #####################################################################################################################

    if not isinstance(dilate_img,np.ndarray):
        dilate_img = tour.DefaultDilateImage(
            img,
            threshold_px    =   threshold_px,
            kernel          =   kernel,
            kernel_area     =   kernel_area,
            is_binary_inv   =   is_binary_inv,
            is_otsu         =   is_otsu,
            is_show         =   is_show
            )
    contours = tour.GetContours_Left2Right(dilate_img,False)

    #####################################################################################################################

    count_segment = 0
    sub_imgs = []
    max_width = ReturnMaxWidth(max_width,img)
    max_height = ReturnMaxHeight(max_height,img)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w>width and h>height and w<max_width and h<max_height:
            img = gray.ColorImage(img)
            sub_img=img[y:y+h, x:x+w].copy()
            show_img = cv2.rectangle(img, (x, y), (x+w, y+h), display_color[count_segment%len(display_color)], 2)
            if is_show==True:
                show.ShowImage(show_img,'Full Image with Columns. '+str(img_title))
                show.ShowImage(sub_img, 'Image with just 1 Column.'+str(img_title))
                count_segment+=1
            sub_imgs.append(sub_img)

    #####################################################################################################################

    if is_show==True:
        print('Number of all Segments:',count_segment)
    if isinstance(img_title,str):
        show.SaveImage(img,img_title,folder=folder,fileformat=fileformat)
        if is_multiple_imgs==True:
            no=0
            for sub_img in sub_imgs:
                if no<10:
                    show.SaveImage(sub_img,img_title+'_0'+str(no),folder=folder,fileformat=fileformat)
                else:
                    show.SaveImage(sub_img,img_title+'_'+str(no),folder=folder,fileformat=fileformat)
                no+=1
    if is_return_img_ls==True:
        return sub_imgs
    return img