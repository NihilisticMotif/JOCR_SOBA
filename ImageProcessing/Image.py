import cv2
import os
import numpy as np 
import GrayImage

class Image:
    def __init__(
            self, 
            img:np.ndarray | str, 
            name:str        =   'Image_FFT', 
            folder:str      =   'Image', 
            fileformat:str  =   'jpg'
            ):
        if type(img) == str:
            self.img = cv2.imread(img)
        elif len(img.shape)==3:
            self.img = np.copy(img)
        elif len(img.shape)==2:
            self.img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        self.name = name
        self.folder = folder           
        self.fileformat = fileformat   

    def Show(self):
        cv2.imshow(self.name,self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Save(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        if self.fileformat[0]=='.':
            self.fileformat = self.fileformat[1:]
        img_path = os.path.join(
            self.folder,self.name+'.'+self.fileformat)
        self.img.save(img_path)

    def Copy(self):
        output_image = Image(
            self.img,
            self.name,
            self.folder,
            self.fileformat)
        return output_image 
    
    def Zoom(
            self, 
            zoom : float = 1, 
            angle: float = 0, 
            coord=None):
        # https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
        # zoom < 1 implies Zoom out
        # zoom > 1 implies Zoom in
        cy, cx = [ i/2 for i in self.img.shape[:-1] ] if coord is None else coord[::-1]
        rotation_matrix = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
        self.img = cv2.warpAffine(self.img, rotation_matrix, self.img.shape[1::-1], flags=cv2.INTER_LINEAR)    

    def RemoveBorders(self):
        contours, heiarchy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
        cnt = cntsSorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        self.img = self.img[y:y+h, x:x+w]

    def CreateBorders(
            self,
            size  : int =50,
            color : list[int] = [255, 255, 255]):
        top, bottom, left, right = [size]*4
        self.img = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    def Crop(self,left:int,width:int,top:int,height:int):
        self.img = self.img[top:top+height, left:left+width]

    def Rotate():
        pass 

    def Gray(self):
        return GrayImage(
            img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY),
            name = self.name,
            folder = self.folder,
            fileformat = self.fileformat
            )
