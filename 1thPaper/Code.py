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

def Sum():
    path='F:/Test/Paper180614/Data/IEMRG/M/'
    outpath='F:/Test/Paper180614/Data/IEMRG/out/'
    arcpy.env.workspace=path
    fs=arcpy.ListRasters()
    for i in range(0,len(fs)-1,3):

        rasS=Raster(fs[i])+Raster(fs[i+1])+Raster(fs[i+2])
        m=fs[i][0:5]+str(int(fs[i][5:7])/3)+'.tif'
        print m
        rasS.save(outpath+m)
    for i in range(1,len(fs),12):
        rasS = Raster(fs[i]) + Raster(fs[i + 1]) + Raster(fs[i + 2]) + Raster(fs[i + 3]) + Raster(fs[i + 4]) + Raster(
            fs[i + 5]) + Raster(fs[i + 6]) + Raster(fs[i + 7]) + Raster(fs[i + 8]) + Raster(fs[i+9]) + Raster(
            fs[i + 10]) + Raster(fs[i + 11])

        m=fs[i][0:5]+'.tif'
        print m
        rasS.save(outpath+m)
# Sum()



def ExtractToPoints():
    path = 'F:/Test/Paper180614/Data/IEMRG/M/'
    Points83='F:/Test/Paper180614/Data/Points/'+'Points83.shp'

    outpath='F:/Test/Paper180614/Data/IEMRG/out/'
    arcpy.env.workspace=path
    fs=arcpy.ListRasters()
    for i in range(0,len(fs),1):

        outshp=outpath+'RG83_'+fs[i][0:len(fs[i])-4]+'.shp'
        print outshp
        ExtractValuesToPoints(Points83, fs[i], outshp, 'INTERPOLATE', "VALUE_ONLY")

# ExtractToPoints()

#     arcpy.AddField_management(rainIMERG, 'M201412', "DOUBLE")
#     for y in range(15,18):
#         for m in range(1,13):
#             fieldName='M'+str((2000+y)*100+m)
#             arcpy.AddField_management(rainIMERG, fieldName, "DOUBLE")
#
#     arcpy.AddField_management(rainIMERG, 'S20144', "DOUBLE")
#     for y in range(15,18):
#         for s in range(1,5):
#             fieldName='S'+str((2000+y)*10+s)
#             arcpy.AddField_management(rainIMERG, fieldName, "DOUBLE")
#
#     for y in range(15, 18):
#         fieldName = 'Y' + str((2000 + y))
#         arcpy.AddField_management(rainIMERG, fieldName, "DOUBLE")

def FieldCombine():

    rainIMERG='F:/Test/Paper180614/Data/IEMRG/'+'RG83_IMERG.shp'
    # path = 'F:/Test/Paper180614/Data/IEMRG/RG83_IMERG.shp/'
    A='rainIMERG'
    arcpy.MakeFeatureLayer_management(rainIMERG, A)

    RGIMERGpath = 'F:/Test/Paper180614/Data/IEMRG/RG83/'
    arcpy.env.workspace=RGIMERGpath
    fs=arcpy.ListFeatureClasses()

    for i in range(0,len(fs),1):

        if (len(fs[i])==16):
            curField='M'+str(fs[i][6:len(fs[i])-4])
            print curField
        elif (len(fs[i])==15):
            curField = 'S' + str(fs[i][6:11])
            print curField
        else:
            curField = 'Y' + str(fs[i][6:10])
            print curField

        B = curField
        arcpy.MakeFeatureLayer_management(fs[i], B)
        arcpy.AddJoin_management(A, "FID", B, "FID")
        arcpy.CalculateField_management(A, 'RG83_IMERG.' + curField, "!" + fs[i][0:len(fs[i]) - 4] + ".RASTERVALU!",
                                        "PYTHON_9.3")
        arcpy.RemoveJoin_management(A)
# FieldCombine()

def CalSta(x,y):
    xave=[sum(x) / len(x) for i in range(0,len(x))]
    yave=[sum(y) / len(y) for i in range(0,len(y))]
    xe= x-xave
    ye= y-yave

    R2 =sum(xe*ye)**2/(sum(xe**2)*sum(ye**2))
    RMSE=np.sqrt(sum((y-x)**2)/len(x))
    Bias=sum(y)/sum(x)-1
    MAE=sum(abs(y-x))/len(x)

    r = []
    r.append(R2)
    r.append(RMSE)
    r.append(Bias)
    r.append(MAE)
    return r
#   GCS计算精度指标
def Sta():
    path='F:/Test/Paper180614/Result/'
    data = xlrd.open_workbook(path+'Validation.xls')

    wb = xlwt.Workbook(encoding='utf-8')

    ws1 = wb.add_sheet('Sta3Mean')  #Sta3Mean
    #   控制计算范围

    t1 = data.sheet_by_index(8)
    t2 = data.sheet_by_index(9)  # 1  2  3取第几个表
    rows=83
    for i in range(0, 51):
        c1 = t1.col_values(i, 0, rows)
        c2 = t2.col_values(i, 0, rows)
        #   全部转浮点，以防万一
        for j in range(0,rows):
            c1[j]=float(c1[j])
            c2[j]=float(c2[j])
        RGS = np.array(c1)
        IMERG= np.array(c2)
        r = CalSta(RGS, IMERG) #注意变量传入位置，应是RGS在前
        for j in range(0, 4):
            ws1.write(j, i, r[j])

    ws1 = wb.add_sheet('1P_Sta3Mean')  #Sta3Mean
    #   控制计算范围

    t1 = data.sheet_by_index(10)
    t2 = data.sheet_by_index(13)  # 1  2  3取第几个表
    rows=31
    for i in range(0, 51):
        c1 = t1.col_values(i, 0, rows)
        c2 = t2.col_values(i, 0, rows)
        #   全部转浮点，以防万一
        for j in range(0,rows):
            c1[j]=float(c1[j])
            c2[j]=float(c2[j])
        RGS = np.array(c1)
        IMERG= np.array(c2)
        r = CalSta(RGS, IMERG) #注意变量传入位置，应是RGS在前
        for j in range(0, 4):
            ws1.write(j, i, r[j])

    ws2 = wb.add_sheet('2P_Sta3Mean')  #Sta3Mean
    #   控制计算范围

    t1 = data.sheet_by_index(11)
    t2 = data.sheet_by_index(14)  # 1  2  3取第几个表
    rows = 21
    for i in range(0, 51):
        c1 = t1.col_values(i, 0, rows)
        c2 = t2.col_values(i, 0, rows)
        #   全部转浮点，以防万一
        for j in range(0, rows):
            c1[j]=float(c1[j])
            c2[j]=float(c2[j])
        RGS = np.array(c1)
        IMERG= np.array(c2)
        r = CalSta(RGS, IMERG) #注意变量传入位置，应是RGS在前
        for j in range(0, 4):
            ws2.write(j, i, r[j])

    ws3 = wb.add_sheet('3P_Sta3Mean')  #Sta3Mean
    #   控制计算范围
    t1 = data.sheet_by_index(12)
    t2 = data.sheet_by_index(15)  # 1  2  3取第几个表
    rows = 31
    for i in range(0, 51):
        c1 = t1.col_values(i, 0, rows)
        c2 = t2.col_values(i, 0, rows)
        #   全部转浮点，以防万一
        for j in range(0, rows):
            c1[j]=float(c1[j])
            c2[j]=float(c2[j])
        RGS = np.array(c1)
        IMERG= np.array(c2)
        r = CalSta(RGS, IMERG) #注意变量传入位置，应是RGS在前
        for j in range(0, 4):
            ws3.write(j, i, r[j])

    out = path + "AllTime_3P_Sta3Mean.xls"       # originIMERGSTA.xls 0.1DIMERGSTA.xls
    wb.save(out)
# Sta()

def fieldSubtract():
    path=''
    rainRes='F:/Test/Paper180614/Temp/'+'rainResidual.shp'
    rainRGS='F:/Test/Paper180614/Temp/'+'RainGauges.shp'
    A = 'rainRes'
    arcpy.MakeFeatureLayer_management(rainRes, A)
    B = 'RainGauges'
    arcpy.MakeFeatureLayer_management(rainRGS, B)
    arcpy.AddJoin_management(A, "FID", B, "FID")

    # arcpy.CalculateField_management(A, 'rainResidual.' + 'M201412','!' +'rainResidual.'+ 'M201412'+'!' +'-'+'!' + 'RainGauges.'+ 'M201412'+'!',
    #                                 "PYTHON_9.3")
    # for y in range(15,18):
    #     for m in range(1,13):
    #         fieldName = 'M' + str((2000 + y) * 100 + m)
    #         arcpy.CalculateField_management(A, 'rainResidual.' + fieldName,
    #                                         '!' + 'rainResidual.' + fieldName + '!' + '-' + '!' + 'RainGauges.' + fieldName + '!',
    #                                         "PYTHON_9.3")

    # for y in range(15,18):
    #     for s in range(1,5):
    #         fieldName='S'+str((2000+y)*10+s)
    #         arcpy.CalculateField_management(A, 'rainResidual.' + fieldName,
    #                                         '!' + 'rainResidual.' + fieldName + '!' + '-' + '!' + 'RainGauges.' + fieldName + '!',
    #                                         "PYTHON_9.3")

    # for y in range(15, 18):
    #     fieldName = 'Y' + str((2000 + y))
    #     arcpy.CalculateField_management(A, 'rainResidual.' + fieldName,
    #                                     '!' + 'rainResidual.' + fieldName + '!' + '-' + '!' + 'RainGauges.' + fieldName + '!',
    #                                     "PYTHON_9.3")

    arcpy.RemoveJoin_management(A)

# fieldSubtract()

