# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *

import sys, string, os
import xlrd
import xlwt
import xlsxwriter
import numpy as np
from scipy.optimize import fminbound
import time
import math
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import *
from dbfread import DBF
arcpy.env.overwriteOutput = True

def main():
    path='F:/Test/Data/RGS/'
    filesPath=path+'PRE/'
    files = os.listdir(filesPath)
    inputFiles = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.txt':# "文件获取"
            inputFiles.append(filesPath + files[i])

    workbook = xlwt.Workbook(encoding='utf-8')

    for i in range(0,len(inputFiles)):    #len(inputFiles)
        name=inputFiles[i][52:58]
        year=int(name)/100
        month= int(name)%100
        # print year,month
        if year==2016 and month==2:
            days=29
        elif month==2:
            days=28
        elif month==4 or month==6 or month==9 or month==11:
            days=30
        else:
            days=31


        dataTxt = np.loadtxt(inputFiles[i])
        # print dataTxt.shape
        rows, cols = dataTxt.shape
        
        dataExcel = np.zeros((1, days + 5), np.float)  #0/1/2/3分别存储X/Y/StationID/Height,
        # 4到days+3存储每天的雨量，days+4存储这个月的月降水量。

        dataTemp = np.zeros((1, days + 5), np.float)    #存储一个站的数据
        for r in range(0, rows):
            # if dataTxt[r,1]>=2830 and dataTxt[r,1]<=3330 and dataTxt[r,2]>=10730 and dataTxt[r,2]<=11630:
            #     dataHB.append(dataTxt[r,:])
            day = int(dataTxt[r, 6])
            if (day == 1):
                dataTemp[0, 0] = int(dataTxt[r, 1]) / 100 + int(dataTxt[r, 1]) % 100 / 60.0
                dataTemp[0, 1] = int(dataTxt[r, 2]) / 100 + int(dataTxt[r, 2]) % 100 / 60.0
                dataTemp[0, 2] = dataTxt[r, 0]
                dataTemp[0, 3] = dataTxt[r, 3] / 10.0
            if (dataTxt[r, 9] < 30000):
                dataTemp[0, 3 + day] = dataTxt[r, 9] / 10.0
                # print day, dataTemp[0, 3 + day]
            elif (dataTxt[r, 9] > 30000 and dataTxt[r, 9] < 31000):
                dataTemp[0, 3 + day] = (dataTxt[r, 9]-30000) / 10.0
            elif (dataTxt[r, 9] > 31000 and dataTxt[r, 9] < 32000):
                dataTemp[0, 3 + day] = (dataTxt[r, 9]-31000) / 10.0
            elif (dataTxt[r, 9] > 32000 and dataTxt[r, 9] < 32700):
                dataTemp[0, 3 + day] = (dataTxt[r, 9]-32000) / 10.0
            else:
                dataTemp[0, 3 + day] = 0

            if (day == days):
                dataTemp[0, 4 + day] = np.sum(dataTemp[0, 4:4 + day])
                dataExcel=np.vstack((dataExcel, dataTemp))
                dataTemp = np.zeros((1, days + 5), np.float)

        rows, cols = dataExcel.shape
        print name,rows-1
        worksheet = workbook.add_sheet(name)     # originIMERG 0.1dIMERG 10kmIMERG
        for r in range(0,rows):#注意跳过第一行的标题行
            for c in range(0,cols):
                worksheet.write(r, c, dataExcel[r,c])

    outFile = path+'Excels/'+'PRE'+".xls"     #  originIMERG.xls  0.1dIMERG.xls    10kmIMERG
    workbook.save(outFile)
    # 'F:/Test/Data/RGS/PRE/SURF_CLI_CHN_MUL_DAY-PRE-13011-201401.txt'
# main()


def month():
    path = 'F:/Test/Data/RGS/'
    filesPath = path + 'PRE/'
    files = os.listdir(filesPath)
    inputFiles = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.txt':  # "文件获取"
            inputFiles.append(filesPath + files[i])

    workbook = xlwt.Workbook(encoding='utf-8')

    for i in range(0, len(inputFiles)):  # len(inputFiles)
        name = inputFiles[i][52:58]
        year = int(name) / 100
        month = int(name) % 100
        # print year,month
        if year == 2016 and month == 2:
            days = 29
        elif month == 2:
            days = 28
        elif month == 4 or month == 6 or month == 9 or month == 11:
            days = 30
        else:
            days = 31

        dataTxt = np.loadtxt(inputFiles[i])
        # print dataTxt.shape
        rows, cols = dataTxt.shape

        dataExcel = np.zeros((1, days + 5), np.float)  # 0/1/2/3分别存储X/Y/StationID/Height,
        # 4到days+3存储每天的雨量，days+4存储这个月的月降水量。

        dataTemp = np.zeros((1, days + 5), np.float)  # 存储一个站的数据
        for r in range(0, rows):
            if dataTxt[r,1]>=2830 and dataTxt[r,1]<=3330 and dataTxt[r,2]>=10730 and dataTxt[r,2]<=11630:
                day = int(dataTxt[r, 6])
                if (day == 1):
                    dataTemp[0, 0] = int(dataTxt[r, 1]) / 100 + int(dataTxt[r, 1]) % 100 / 60.0
                    dataTemp[0, 1] = int(dataTxt[r, 2]) / 100 + int(dataTxt[r, 2]) % 100 / 60.0
                    dataTemp[0, 2] = dataTxt[r, 0]
                    dataTemp[0, 3] = dataTxt[r, 3] / 10.0
                if (dataTxt[r, 9] < 30000):
                    dataTemp[0, 3 + day] = dataTxt[r, 9] / 10.0
                    # print day, dataTemp[0, 3 + day]
                elif (dataTxt[r, 9] > 30000 and dataTxt[r, 9] < 31000):
                    dataTemp[0, 3 + day] = (dataTxt[r, 9] - 30000) / 10.0
                elif (dataTxt[r, 9] > 31000 and dataTxt[r, 9] < 32000):
                    dataTemp[0, 3 + day] = (dataTxt[r, 9] - 31000) / 10.0
                elif (dataTxt[r, 9] > 32000 and dataTxt[r, 9] < 32700):
                    dataTemp[0, 3 + day] = (dataTxt[r, 9] - 32000) / 10.0
                else:
                    dataTemp[0, 3 + day] = 0

                if (day == days):
                    dataTemp[0, 4 + day] = np.sum(dataTemp[0, 4:4 + day])
                    dataExcel = np.vstack((dataExcel, dataTemp))
                    dataTemp = np.zeros((1, days + 5), np.float)
            else:
                continue
        rows, cols = dataExcel.shape
        print name, rows - 1
        worksheet = workbook.add_sheet(name)  # originIMERG 0.1dIMERG 10kmIMERG
        for r in range(0, rows):  # 注意跳过第一行的标题行
            for c in range(0, cols):
                worksheet.write(r, c, dataExcel[r, c])

    outFile = path + 'Excels/' + 'PRE' + ".xls"  # originIMERG.xls  0.1dIMERG.xls    10kmIMERG
    workbook.save(outFile)
    # 'F:/Test/Data/RGS/PRE/SURF_CLI_CHN_MUL_DAY-PRE-13011-201401.txt'
# month()

def monthHB():
    path = 'F:/Test/Data/RGS/'
    fileID=path+'Excels/'+'RGS36_HB'+'.xls'
    excelID=xlrd.open_workbook(fileID)
    tableID=excelID.sheet_by_index(0)
    rows=tableID.nrows
    ID = tableID.col_values(4, 1, rows)

    filePre=path+'Excels/'+'PRE_Day_BiggerHB'+'.xls'
    excelPre = xlrd.open_workbook(filePre)
    ts=excelPre.nsheets
    # print ts
    dataHB=[]
    for i in range(12,ts):
        k=0
        data=[]
        tPre = excelPre.sheet_by_index(i)

        rows ,cols= tPre.nrows,tPre.ncols

        for r in range(1,rows):
            if tPre.cell_value(r,2)==ID[k]:
                dPre=tPre.cell_value(r,cols-1)
                data.append(dPre)
                k=k+1
                if (k==32):
                    break
        dataHB.append(data)

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('monthPreHB')  # originIMERG 0.1dIMERG 10kmIMERG

    for r in range(0, 32):  # 注意跳过第一行的标题行
        for c in range(0, 36):
            worksheet.write(r, c, dataHB[c][r])

    outFile = path + 'Excels/' + '1monthPreHB' + ".xls"  # originIMERG.xls  0.1dIMERG.xls    10kmIMERG
    workbook.save(outFile)
# monthHB()