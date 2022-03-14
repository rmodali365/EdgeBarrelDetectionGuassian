from concurrent.futures import thread
from gettext import npgettext
from typing import final
from xml.etree.ElementTree import PI
import cv2
from cv2 import getGaussianKernel
from cv2 import threshold
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image, ImageSequence
import math as m
import sys
from skimage import data, io, filters as sk

# RGB color reader orange
# colorImg = cv2.imread("Frame0064.png")
# orangeFilter = cv2.inRange(colorImg,(209, 122, 9), (255,165,0))
# cv2.imshow("orange",orangeFilter)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# HSV color reader orange
# img = cv2.imread("Frame0064.png")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv,(10, 100, 20), (25, 255, 255) )
# cv2.imshow("orange", mask)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 4.1
# cap = cv2.VideoCapture('Vid.mp4')
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:

#         threshed = cv2.inRange(frame,(10, 100, 20), (25, 255, 255))
#         # newFrame = np.array(threshed)

#         cv2.imshow('newFrame',threshed)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

img = cv2.imread("Frame0064.png")
inputMask = cv2.imread("Label0064.png")
maskOutput = cv2.bitwise_and(img,inputMask)
means,devs = cv2.meanStdDev(maskOutput,mask=inputMask[:,:,0])
print('means: ',means)
print("devs: ",devs)
totalPix = 76800
diff = np.zeros((3,1))

cap = cv2.VideoCapture('Vid.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        img2 = Image.fromarray(frame)
        # img2 = PIL.Image.open(frame)
        
        pix = img2.load()
        sumSigma = np.zeros((3,3))
        for y in range(0,240):
            for x in range(0,320):
                pixel = pix[x,y]
                x_i = [[pixel[0]],[pixel[1]],[pixel[2]]]
                diff = np.subtract(x_i,means)
                total = np.dot(diff,np.transpose(diff))
                sumSigma+=total
        finalSigma = sumSigma/totalPix
        # print(finalSigma)
        for y in range(0,240):
            for x in range(0,320):
                pixel2 = pix[x,y]
                x_i2 = [[pixel2[0]],[pixel2[1]],[pixel2[2]]]
                diff2 = np.subtract(x_i2,means)
                calc1 = np.dot(np.transpose(diff2),np.linalg.inv(finalSigma))
                calc2 = -1/2*np.dot(calc1,diff2)
                finalCertainty = m.pow(m.e,calc2)
                print(finalCertainty)
                if(finalCertainty<=0.5):
                    value = (0,0,0)
                    img2.putpixel((x, y), value)
                else:
                    value = (255,255,255)
                    img2.putpixel((x, y), value)
        img2arr = np.array(img2,dtype=float)
        # plt.imshow(img2arr, interpolation='nearest')
        # plt.show()
        cv2.imshow("gaussian", img2arr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

