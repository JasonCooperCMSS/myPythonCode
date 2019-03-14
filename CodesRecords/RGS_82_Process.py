# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *

import sys, string, os
import xlrd, xlwt, xlsxwriter
import numpy as np
import pandas as pd
from scipy.optimize import fminbound
import time
import math
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import *
from dbfread import DBF

arcpy.env.overwriteOutput = True


def DataCheck():
    path = "F:/Test/Temp/RGS_82_Process/"
    filesPath = path + 'DailyRainfall/'
    files = os.listdir(filesPath)
    inputFiles = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.xls':  # "文件获取"
            inputFiles.append(filesPath + files[i])
    count = np.ones((2, len(files)), np.int)
    for i in range(0, len(inputFiles)):
        df = pd.read_excel(inputFiles[i])
        count[0, i] = df.shape[0]
        count[1, i] = df.shape[1]
    # print count
    outFile = "F:/Test/Temp/RGS_82_Process/" + u"数据范围初步验证.xlsx"
    df = pd.DataFrame(count)
    df.to_excel(outFile, header=True, index=False, encoding='utf-8')
    print df.head()


# DataCheck()

def DataMerge():
    path = "F:/Test/Temp/RGS_82_Process/"
    filesPath = path + 'DailyRainfall/'
    files = os.listdir(filesPath)
    inputFiles = []
    Index = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.xls':  # "文件获取"
            # print os.path.splitext(files[i])[0]
            Index.append(os.path.splitext(files[i])[0])
            inputFiles.append(filesPath + files[i])
    # print Index
    arrData = np.zeros((1840, 82), np.float)
    arrTemp = np.zeros((1840, 1), np.float)
    for i in range(0, len(inputFiles)):
        tempDf = pd.read_excel(inputFiles[i])
        arrDf = np.array(tempDf.as_matrix()[:, -2:], np.float)
        if i == 48:
            print arrDf[200, 0]
            print arrDf.shape[0]
        if arrDf.shape[0] < 1840:
            print '数据缺失 !'
            arrTemp[0:365, 0] = [999990 for z in range(0, 365)]
            # print arrTemp[0:365, 0]
            start = 365
        else:
            start = 0
        for j in range(0, arrDf.shape[0]):
            if arrDf[j, 0] < 10000:
                arrTemp[start, 0] = arrDf[j, 0]
                start = start + 1
                continue
            elif arrDf[j, 1] < 10000:
                arrTemp[start, 0] = arrDf[j, 1]
                start = start + 1
                continue
            else:
                arrTemp[start, 0] = 999990
                start = start + 1
        # print arrTemp[0,0]
        # print arrTemp.shape
        arrData[:, i:i + 1] = arrTemp
    dfData = pd.DataFrame(arrData, columns=Index)
    print dfData['57484'].head()
    outFile = "F:/Test/Temp/RGS_82_Process/" + u"初次合并结果.xlsx"
    dfData.to_excel(outFile, header=True, index=False, encoding='utf-8')


# DataMerge()

def DataMissAna():
    path = "F:/Test/Temp/RGS_82_Process/" + u"初次合并结果.xlsx"
    df = pd.read_excel(path, header=0)
    arr = np.array(df.as_matrix()[729:], np.float)
    count = []
    for c in range(0, arr.shape[1]):
        countc = [0 for i in range(0, 5)]
        count1, count2, count5, count15, count30 = 0, 0, 0, 0, 0
        r = 0
        while r < arr.shape[0]:
            if arr[r, c] > 10000:
                for t2 in range(r, arr.shape[0]):
                    if arr[t2, c] < 10000:
                        numc = t2 - r
                        r = t2
                        break
                    else:
                        continue
                # if c==48:
                #     print numc
                if numc >= 30:
                    count30 = count30 + 1
                elif numc >= 15 and numc < 30:
                    count15 = count15 + 1
                elif numc >= 5 and numc < 15:
                    count5 = count5 + 1
                elif numc >= 2 and numc < 5:
                    count2 = count2 + 1
                else:
                    count1 = count1 + 1
            else:
                r = r + 1
        countc[0], countc[1], countc[2], countc[3], countc[4] = count1, count2, count5, count15, count30
        count.append(countc)
    outDf = pd.DataFrame(count, index=df.columns, columns=['count1', 'count2', 'count5', 'count15', 'count30'])
    print outDf.head()
    outFile = "F:/Test/Temp/RGS_82_Process/" + u"缺失情况详细检查1.xlsx"
    outDf.to_excel(outFile, header=True, index=True, encoding='utf-8')
# DataMissAna()
