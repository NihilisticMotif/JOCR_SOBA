

def OddKernelArea(num):
    num=int(num)
    if num<3:
        return 3 
    else:
        if num%2==1:
            return num 
        else:
            return num+1

def SetPx(num):
    num=int(num)
    if num < 0:
        return 0 
    elif num > 255:
        return 255
    else:
        return num

def ReturnValidInput(input,input_options,message):
    if input not in input_options:
        print(message)
        return input_options[0]
    else:
        return input

'''
Reference
1. Tesseract OCR Tutorial
* https://nanonets.com/blog/ocr-with-tesseract/
2. How to Open an Image in Python with PIL
* https://youtu.be/UxYJxcdLrs0?si=biuELY46TWTwhqvX

'''