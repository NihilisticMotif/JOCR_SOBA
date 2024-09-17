import numpy as np
import cv2
import GrayImage
from Image import Image
from typing import List
class MarkImg:
    def __init__(self,
            img:Image,
            dilate_img:GrayImage):
        self.img:Image = img
        self.mark_img:Image = img.copy()
        self.sub_img:List[Image] = []   # Module cannot be used as a type
        # ...
        self.dilate_img:GrayImage = dilate_img
        self.contour:list = self.__CreateContour()
        self.select_contour:list = []
    
    def __CreateContour(self):
        contours, hierarchy = cv2.findContours(self.dilate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def __SortContour(self,is_descending:bool=False,sort_method:str='size'):
        left2right = ['left2right','lefttoright']
        top2button = ['top2button','toptobutton','up2down','uptodown']
        width = ['width','widthsize']
        height = ['height','heightsize']
        area = ['size','area']
        sort_method  = sort_method.replace(" ", "").lower()
        if sort_method in left2right:
            sort_method = 0
        elif sort_method in top2button:
            sort_method = 1
        elif sort_method in width:
            sort_method = 2
        elif sort_method in height:
            sort_method = 3
        elif sort_method in area:
            sort_method = -1
        else:
            sort_method = -1
        if sort_method in [0,1,2,3]:
            self.contour = sorted(self.contour,key=lambda x:cv2.boundingRect(x)[sort_method],reverse=is_descending)
        else:
            self.contour = sorted(self.contour,key=lambda x:cv2.contourArea,reverse=is_descending)

    def SelectContour(self,
            display_color:list[list[int]],
            min_width:int,
            max_width:int|None,
            min_height:int,
            max_height:int|None,
            sort_method:str='lefttoright',
            is_descending:bool=False):
        self.__SortContour(is_descending,sort_method)
        self.select_contour = []
        self.mark_img = self.img
        self.sub_img = []
        color_mod = 0
        if max_width == None:
            max_width:int = self.img.shape[0]
        if max_height == None:
            max_height:int = self.img.shape[1]
        for contour in self.contour:
            x, y, w, h = cv2.boundingRect(contour)
            if w>min_width and h>min_height and w<max_width and h<max_height:
                self.select_contour.append(contour)
                self.mark_img = cv2.rectangle(self.mark_img, (x, y), (x+w, y+h), display_color[color_mod%len(display_color)], 2)
                color_mod += 1
                sub_img=self.mark_img[y:y+h, x:x+w].copy()
                self.sub_img.append(sub_img)

    def SaveAllSubImages(self,
            name:str='SubImg',
            folder:str='SubImg',
            fileformat:str='jpg'):
        index=0
        for img in self.sub_img:
            img.name = name+'_'+str(index)
            img.folder = folder
            img.fileformat = fileformat
            index+=1
            img.Save()

    def ShowAllSubImage(self):
        index=0
        for img in self.sub_img:
            img.name = img.name.split('_')[0]
            img.name = img.name+'_'+str(index)
            img.Show()
    
    @property 
    def UpdateContour(self):
        self.contour = self.__CreateContour()
        self.select_contour = []


