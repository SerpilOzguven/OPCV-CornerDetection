# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 22:37:44 2022

@author: Serpil ÖZGÜVEN
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktar
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")


#harris corner detection
#blocksize komşuluk boyutu
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

#genişletme yapalım
dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

#farklı bir kenar algılama yöntemi ile devam edelim
#shi thomsai algoritması ile detection yapalım
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)
#120tane köşe tespit der. 10 yazsak 10 tanesini tespit eder.
# çalıtıralım ve corners'ın veritipini değiştirelim
corners = np.int64(corners)


for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3,(125,125,125), cv2.FILLED)
    
    
plt.imshow(img)
plt.axis("off")    













