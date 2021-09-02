import cv2
import imutils
import numpy as np
image = cv2.imread('images/Cars1.png')
image = cv2.resize(image,(600,400))
BandW = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
BandW = cv2.bilateralFilter(BandW,8,20,10)
edges = cv2.Canny(BandW,20,200)

contours = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c,True)
    app = cv2.approxPolyDP(c,0.018*peri, True)
    if len(app) == 4:
        screenCnt = app
        break
mask = np.zeros(BandW.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.imwrite('fin.png',new_image)
