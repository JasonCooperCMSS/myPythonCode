# coding:gbk
import sys
import os,math,random
import arcpy
import numpy as np
import pandas as pd

def func_1(step=8,winWidth=15,winHeight=15,minPre=0.1,overlapThed=20,corrThed=0.5):

    filein_L = os.path.join(os.getcwd(), r'C:\\Users\\ASUS\\Desktop\\LiuMin\\QPE_20190621_HBAL.tif')
    filein_R = os.path.join(os.getcwd(),r'C:\\Users\\ASUS\\Desktop\\LiuMin\\Resampled_IMERGD_20190621_HBAL.tif')
    Raster_L = arcpy.Raster(filein_L)
    Raster_R = arcpy.Raster(filein_R)
    arr_L = arcpy.RasterToNumPyArray(Raster_L, nodata_to_value=0)
    arr_R = arcpy.RasterToNumPyArray(Raster_R, nodata_to_value=0)

    '''以20*20窗口，步长10滚动'''
    nrows_L, ncols_L = arr_L.shape[0],arr_L.shape[1]
    nrows_R, ncols_R = arr_R.shape[0],arr_R.shape[1]
    n1 = 0   #  起始窗口的左上角行号
    n2 = 0   #  起始窗口的左上角列号
    arrList_L = []
    arrList_R = []
    while (n1 + winWidth <= nrows_L):
        while (n2 + winHeight <= ncols_L):
            b = arr_L[n1:n1 + winWidth:1, n2:n2 + winHeight:1]
            n2 += step
            arrList_L.append(b)
        n1 += step
        n2 = 0
    # print(len(arrList_L))
    n1 = 0   #  起始窗口的左上角行号
    n2 = 0   #  起始窗口的左上角列号
    while (n1 + winWidth <= nrows_R):
        while (n2 + winHeight <= ncols_R):
            c = arr_R[n1:n1 + winWidth:1, n2:n2 + winHeight:1]
            n2 += step
            arrList_R.append(c)
        n1 += step
        n2 = 0
    # print(arrList_R)
    overpre = True
    num_zeros_R=0
    for i in arrList_R:
        for g in range(0, i.shape[0]):
            for j in range(0, i.shape[1]):
                while (j<0.1):
                    num_zeros_R +=1
        arr_size_R = winHeight*winWidth
        zero_pre= num_zeros_R / arr_size_R
        print(zero_pre)
        if zero_pre < minPre:
            overpre = False
            break
    if overpre == True:
        arrList_R.remove(i)
    print(len(arrList_R))

    return 0

def main():
    func_1()
    return 'OK!'
main()




