# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:03:43 2019

@author: kentlohr
"""
import numpy as np
import cv2

def bilateralFilter(img, frame, sigmas, sigmar):
    B, G, R = cv2.split(img)
    B_tran, G_tran, R_tran = cv2.split(img)
    m, n = B.shape
    #print(m, n)
    s_square = 2 * sigmas * sigmas
    r_square = 2 * sigmar * sigmar
    radius = int(frame/2)
    for row in range(m):
        for col in range(n):
            value_B = 0
            weight_B = 0
            value_G = 0
            weight_G = 0
            value_R = 0
            weight_R = 0
            for k in range(row-radius, row+radius+1):
                for l in range(col-radius, col+radius+1):
                    if((k<0)or(l<0)or(k>=m)or(l>=n)):
                        val_B = 0
                        val_G = 0
                        val_R = 0
                    else:
                        val_B = B[k][l]
                        val_G = G[k][l]
                        val_R = R[k][l]
                    gauss_value_B = gaussW(row, col, k, l, B[row][col], val_B, s_square, r_square)
                    value_B += val_B * gauss_value_B
                    weight_B += gauss_value_B
                    gauss_value_G = gaussW(row, col, k, l, G[row][col], val_G, s_square, r_square)
                    value_G += val_G * gauss_value_G
                    weight_G += gauss_value_G
                    gauss_value_R = gaussW(row, col, k, l, R[row][col], val_R, s_square, r_square)
                    value_R += val_R * gauss_value_R
                    weight_R += gauss_value_R
            B_tran[row][col] = np.uint8(value_B/weight_B)
            G_tran[row][col] = np.uint8(value_G/weight_G)
            R_tran[row][col] = np.uint8(value_R/weight_R)
    print('finished.')
    img_tran = cv2.merge([B_tran, G_tran, R_tran])
    #cv2.imshow("after", img_tran)
    cv2.imwrite("after.jpg", img_tran)

def gaussW(i, j, k, l, color_value, area_value, s, r):
    space = (np.square(i-k) + np.square(j-l)) / s
    color = np.square(color_value-area_value) / r
    w = np.exp(-space-color)
    return w
    
def main():
    img = cv2.imread("mopi.png")
    #cv2.imshow("original image", img)
    bilateralFilter(img, 7, 2, 25)
    #dst = cv2.bilateralFilter(src=img, d=0, sigmaColor=50, sigmaSpace=15)
    #dst = cv2.fastNlMeansDenoisingColored(img, templateWindowSize = 7, searchWindowSize = 21, h=5)
    #dst = cv2.GaussianBlur(img, (7,7), 50)
    #cv2.imwrite("after1.jpg", dst)

if __name__ == '__main__':
    main()
    