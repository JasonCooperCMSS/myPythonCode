# coding:utf-8
import arcpy
from arcpy import env

import sys, string, os
import xlrd
import xlwt
import xlsxwriter

from scipy.optimize import fminbound
import time
import math
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import *
from dbfread import DBF



import pandas as pd
import numpy as np
arcpy.env.overwriteOutput = True

def func_1(inFile="", outFile=""):  # 3小时地面观测站数据，合并成日度数据，以17-20共8个数值相加，作为当天降水量
    inPath = "F:/Test/Paper180614/Data/PreExtreme"
    inFile = os.path.join(inPath, "OriginData.csv") #SK_323_201801_06
    outFile = os.path.join(inPath, "TempExtreme.csv")
    inDf = pd.read_csv(inFile, header=0, index_col=0)

    nrows,ncols=inDf.shape[0],inDf.shape[1]
    # if nrows !=488 or ncols!=82:

    #     raise ValueError("Origin data was wrong !")
    index,names=inDf.index,inDf.columns
    outDf = pd.DataFrame(index=inDf.columns)
    print(names)
    newIndex=[]
    btime, etime = 0, 8
    for i in range(0,nrows):
        if index[i]%100==11:
            btime=i
        if index[i]%100==8:
            etime=i+1
            newIndex.append(index[i]/100)
            # print(inDf[:][btime:etime].mean())
            # break
            outDf=pd.concat((outDf,inDf[:][btime:etime].sum()),axis=1)
            # break
    outDf.columns=newIndex
    outDf.index.name = 'RGS'
    print(outDf)
    outDf.to_csv(outFile)
    return 0



def func_2():#nc转tif,裁剪，值提取到点
    inPath="F:/Test/Data/IMERG/IMERGDV6_nc_20180601_20180731"
    outPath="F:/Test/Data/IMERG/IMERGDV6_tif_20180601_20180731"
    Points323="F:/Test/Paper180614/Data/Points/"+"Points323.shp"
    shpPath="F:/Test/Paper180614/Data/IEMRG/RGS323_DAY"
    Excel="F:/Test/Paper180614/Data/PreExtreme/"+"IMERGPreExtreme.csv"

    # arcpy.env.workspace = outPath
    Df=pd.DataFrame()

    for file in os.listdir(inPath):
        # 3B-DAY.MS.MRG.3IMERG.20180609-S000000-E235959.V06.nc4.nc
        fileName=file[21:29]
        print(fileName)
        ncFile=os.path.join(inPath,file)
        arcpy.MakeNetCDFRasterLayer_md(ncFile, "precipitationCal", "lon", "lat",out_raster_layer=fileName)
        tifFile = os.path.join(outPath,'I'+fileName + '.tif')
        arcpy.CopyRaster_management(fileName, out_rasterdataset=tifFile, format="TIFF")

        outshp = os.path.join(shpPath,'RGS323_I'+fileName + '.shp')
        arcpy.sa.ExtractValuesToPoints(Points323, tifFile, outshp, 'INTERPOLATE', "VALUE_ONLY")

        tempList = arcpy.da.FeatureClassToNumPyArray(outshp, ('RASTERVALU'))
        tempDf=pd.DataFrame(data=tempList['RASTERVALU'],columns=[fileName])
        Df=pd.concat((Df,tempDf),axis=1)
        # print(Df)
        if fileName=='20180602':
            # break
            continue
    Df.to_csv(Excel)
    return 0


def func_3():
    inPath="F:/Test/Data/IMERG/06D/"
    arcpy.env.workspace = inPath
    tifFiles=arcpy.ListRasters("*", "TIF")
    out="F:/Test/Paper180614/Data/PreExtreme/Ras/tif/IMV6D_20180616_0716_Plus.tif"
    outRas=arcpy.Raster(tifFiles[0])
    for i in range(1,len(tifFiles)):
        print(tifFiles[i])
        outRas=outRas+arcpy.Raster(tifFiles[i])
    outRas.save(out)


def StaAna():
    def CalSta(x,y):
        n = len(x)
        xsum, ysum = np.sum(x), np.sum(y)
        xave = [xsum / n for i in range(0, n)]
        yave = [ysum / n for i in range(0, n)]
        xe, ye = x - xave, y - yave
        r = np.sum(xe * ye) / np.sqrt(sum(xe *xe) * sum(ye *ye))
        rmse = np.sqrt(np.sum((y - x) ** 2) / n)
        bias = ysum / xsum - 1
        mae = np.sum(np.abs(y - x)) / n
        result = [r, rmse, bias, mae]
        return result


    inFile="F:/Test/Paper180614/Data/MS_StaAna/"+"MS_StaAna.xls"
    outCsv="F:/Test/Paper180614/Data/MS_StaAna/"+"M_SpaceAnalysis.xls"

    xDf=pd.read_excel(inFile,sheetname=0,header=0, index_col=0)
    yDf=pd.read_excel(inFile,sheetname=1,header=0, index_col=0)

    nrows,ncols=xDf.shape[0],xDf.shape[1]
    print(nrows,ncols)

    results = []
    cList =xDf.columns
    for c in cList:
        x,y=xDf[c].values,yDf[c].values
        print(x,y)
        result=CalSta(x,y)
        # print(result)
        results.append(result)
        # break
    # print(results)
    outDf=pd.DataFrame(results,columns=['R','RMSE','Bias','MAE'],index=cList)
    print(outDf)
    outDf.to_excel(outCsv)
    return results

    # inFile="F:/Test/Paper180614/Data/PreExtreme/"+"StaAnalysis.xls"
    # outCsv="F:/Test/Paper180614/Data/PreExtreme/"+"TimeAnalysis_V2.xls"
    #
    # xDf=pd.read_excel(inFile,sheetname=2,header=0, index_col=0)
    # yDf=pd.read_excel(inFile,sheetname=3,header=0, index_col=0)
    #
    # nrows,ncols=xDf.shape[0],xDf.shape[1]
    # print(nrows,ncols)
    # print()
    # cList=[]
    # for c in xDf.columns:
    #     cList.append(str(c))
    # xDf.columns,yDf.columns=cList,cList
    #
    # results = []
    # cList = ['57494','57256','57279','57381','57399','58409','57582','57476','57483','57377','57358','57439']
    # for c in cList:
    #     x,y=xDf[c].values,yDf[c].values
    #     print(x,y)
    #     result=CalSta(x,y)
    #     # print(result)
    #     results.append(result)
    #     # break
    # # print(results)
    # outDf=pd.DataFrame(results,columns=['R','RMSE','Bias','MAE'],index=cList)
    # print(outDf)
    # outDf.to_excel(outCsv)

def func_4():
    def CalSta(x,y):
        n = len(x)
        xsum, ysum = np.sum(x), np.sum(y)
        xave = [xsum / n for i in range(0, n)]
        yave = [ysum / n for i in range(0, n)]
        xe, ye = x - xave, y - yave
        r = np.sum(xe * ye) / np.sqrt(sum(xe *xe) * sum(ye *ye))
        rmse = np.sqrt(np.sum((y - x) ** 2) / n)
        bias = ysum / xsum - 1
        mae = np.sum(np.abs(y - x)) / n
        result = [r, rmse, bias, mae]
        return result


    inFile="F:/Test/Paper180614/Data/PreExtreme/"+"DataRainyAndExtreme.xls"
    outCsv="F:/Test/Paper180614/Data/PreExtreme/"+"Res_RainyAndExtreme.xls"

    xDf=pd.read_excel(inFile,sheetname=0,header=0, index_col=0)
    yDf=pd.read_excel(inFile,sheetname=1,header=0, index_col=0)

    nrows,ncols=xDf.shape[0],xDf.shape[1]
    print(nrows,ncols)

    results = []
    preRank=[1,10,25,50]
    # cList =xDf.columns
    # n=nrows*ncols
    # for rank in preRank:
    #     right,fail,fake=0,0,0
    #     for c in range(ncols):
    #         for r in range(nrows):
    #             x, y = xDf.iloc[r,c], yDf.iloc[r,c]
    #             # print(x, y)
    #             if (x>=rank and y>=rank) or (x<rank and y<rank):
    #                 right+=1
    #             elif (x>=rank and y<rank):
    #                 fail+=1
    #             else:
    #                 fake+=1
    #     n=float(n)
    #     result = [n,right,fail,fake,right/n,fail/n,fake/n]
    #     results.append(result)

    cList =xDf.columns
    n=nrows*ncols

    countList=[[0,0,0,0,0] for i in range(5)]
    for c in range(ncols):
        for r in range(nrows):
            x, y = xDf.iloc[r,c], yDf.iloc[r,c]
            # print(x, y)
            if (x<1):
                if (y<1):
                    countList[0][0]+=1
                elif (y>=1 and y <10):
                    countList[0][1] += 1
                elif (y>=10 and y <25):
                    countList[0][2] += 1
                elif (y>=25 and y <50):
                    countList[0][3] += 1
                elif (y>=50):
                    countList[0][4] += 1
            elif x>=1 and x<10:
                if (y<1):
                    countList[1][0]+=1
                elif (y>=1 and y <10):
                    countList[1][1] += 1
                elif (y>=10 and y <25):
                    countList[1][2] += 1
                elif (y>=25 and y <50):
                    countList[1][3] += 1
                elif (y>=50):
                    countList[1][4] += 1
            elif x>=10 and x<25:
                if (y<1):
                    countList[2][0]+=1
                elif (y>=1 and y <10):
                    countList[2][1] += 1
                elif (y>=10 and y <25):
                    countList[2][2] += 1
                elif (y>=25 and y <50):
                    countList[2][3] += 1
                elif (y>=50):
                    countList[2][4] += 1
            elif x>=25 and x<50:
                if (y<1):
                    countList[3][0]+=1
                elif (y>=1 and y <10):
                    countList[3][1] += 1
                elif (y>=10 and y <25):
                    countList[3][2] += 1
                elif (y>=25 and y <50):
                    countList[3][3] += 1
                elif (y>=50):
                    countList[3][4] += 1
            elif x>=50 :
                if (y<1):
                    countList[4][0]+=1
                elif (y>=1 and y <10):
                    countList[4][1] += 1
                elif (y>=10 and y <25):
                    countList[4][2] += 1
                elif (y>=25 and y <50):
                    countList[4][3] += 1
                elif (y>=50):
                    countList[4][4] += 1

        #
        # n=float(n)
        # result = [n,right,fail,fake,right/n,fail/n,fake/n]
        # results.append(result)
    print(countList)


def main():
    # func_1()
    # func_2()
    # func_3()
    # StaAna()
    func_4()
    return 0
main()
