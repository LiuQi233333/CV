# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:46:42 2019

@author: kentlohr
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Harris(img):
    # Harris角点检测基于灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Harris角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # 腐蚀一下，便于标记
    dst = cv2.dilate(dst, None)
    # 角点标记为红色
    img[dst > 0.0001 * dst.max()] = [0, 255, 0]
    cv2.imwrite('blox-RedPoint.png', img)
    cv2.imshow('dst', img)
    cv2.waitKey(0)

def Fast(img):
    # 初始化FAST特征检测函数
    fast = cv2.FastFeatureDetector_create()
    
    # 找到并画出关键点
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(0,255,0))
    
    # 输出参数
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    
    cv2.imshow('fast_true',img2)
    
    # 不用非最大值抑制的情况
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img,None)
    
    print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
    
    img3 = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
    
    cv2.imshow('fast_false',img3)
    
    cv2.waitKey(0)

def MSER(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create(_min_area=300)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)
    
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0), 2)
    
    plt.imshow(img,'brg')
    plt.show()

def DoG(img):
     # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建一个sift对象 并计算灰度图像
    sift = cv2.xfeatures2d.SIFT_create(400)
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    
    # 在图像上绘制关键点
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS表示对每个关键点画出圆圈和方向
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))
    
    cv2.imshow("sift_keypoints", img)
    cv2.waitKey(0)

def main():
    img = cv2.imread('img3.ppm')
    Harris(img)

if __name__ == '__main__':
    main()