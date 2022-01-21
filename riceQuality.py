import numpy as np
import cv2

def getResults(img):

    Slender = 0
    Medium = 0
    Bold = 0
    Round = 0

    img = cv2.imread(img,0)
    ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.float32)/9
    dst = cv2.filter2D(binary,-1,kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(dst,kernel2,iterations = 1)
    dilation = cv2.dilate(erosion,kernel2,iterations = 1)
    edges = cv2.Canny(dilation,100,200)
    contours,hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    totalAR=0
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        aspectRatio = float(w)/h
        if(aspectRatio<1):
            aspectRatio=1/aspectRatio
        
        aspectRatio = round(aspectRatio,1)
        if(aspectRatio>=3):
            Slender=Slender+1
        elif(aspectRatio>=2.1 and aspectRatio<3):
            Medium=Medium+1
        elif(aspectRatio>=1.1 and aspectRatio<2.1):
            Bold=Bold+1
        elif(aspectRatio<=1):
            Round=Round+1

        totalAR+=aspectRatio

    return Slender, Medium, Bold, Round, len(contours)

