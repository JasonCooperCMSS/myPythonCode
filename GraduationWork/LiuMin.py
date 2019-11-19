# coding:utf-8
import os,shutil,time,math,random
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td

# from arcpy import env
# arcpy.CheckOutExtension("spatial")
# arcpy.env.overwriteOutput = True
"dwwwwwwwwwwwww"
# 临时试探代码
def temp():
    inPath = "F:/Test/GraduationWork/Data/Temp"
    inFile = os.path.join(inPath, "TempDaily.csv") #SK_323_201801_06
    outFile = os.path.join(inPath, "TempDaily.csv")
    inDf = pd.read_csv(inFile, header=0, index_col=0)
    nrows,ncols=inDf.shape[0],inDf.shape[1]
    # if nrows !=488 or ncols!=82:
    #     raise ValueError("Origin data was wrong !")
    index,names=inDf.index,inDf.columns
    newIndex=[]
    print(inDf.index)

# 根据每列数据计数，把数据最少的前N列/站删除，count非零行数量，然后排序
def func_2(inFile="", outFile="", numToDel=20):  #
    inPath = "F:/Test/GraduationWork/Data/Temp"
    inFile = os.path.join(inPath, "SK_323_20190802.csv") #SK_323_201801_06
    outFile = os.path.join(inPath, "TempDeleteNcols.csv")
    inDf = pd.read_csv(inFile, header=0, index_col=0)

    nrows,ncols=inDf.shape[0],inDf.shape[1]
    # if nrows !=488 or ncols!=82:
    #     raise ValueError("Origin data was wrong !")
    index,names=inDf.index,inDf.columns
    # print(inDf.count(axis=0))
    countList=inDf.count(axis=0).values.tolist()
    for i in range(0,numToDel):
        colToDel=inDf.count(axis=0).idxmin().values.tolist()
        for c in colToDel:
            # print(colToDel)
            inDf=inDf.drop(str(c),axis=1)
        # break
    # countList = inDf.count(axis=0).values.tolist()
    # print(countList)

    inDf.index.name = 'RGS'
    print(inDf.head(5))
    print(inDf.tail(5))
    inDf.to_csv(outFile)
    return 0

# 根据总的范围，输入数据大小，允许重合度，推算在2019年7个月时间里能组成多少个雨日样本。
# 以经纬度计算，不考虑优化。以满足最小降水阈值的离（0,90）最近点开始，雨日标准/窗口大小/重叠程度
def func_3():
    inPath = "F:/Test/GraduationWork/Data/Temp"
    pointsFile = os.path.join(inPath, "RGS82Points.xls")  # SK_323_201801_06
    rgssFile=os.path.join(inPath, "TempDaily_2019_82RGS.csv")
    outFile = os.path.join(inPath, "outInfos.csv")


    preThed=0.5
    winWidth,winHeight=3,3
    overlapThred=0.5

    pointsDf = pd.read_excel(pointsFile, header=0, index_col=0)
    pidx,pnames=pointsDf.index,pointsDf.columns
    prows,pcols=pointsDf.shape[0],pointsDf.shape[1]
    rgssDf=pd.read_csv(rgssFile, header=0, index_col=0)
    ridx, rnames = rgssDf.index, rgssDf.columns
    rrows, rcols = rgssDf.shape[0], rgssDf.shape[1]

    "----------------------------------------------------"
    "--------------------------------------------------"
    allSampleNum,allSampleList=0,[]
    for i in ridx:
        tempDf=rgssDf.loc[i,:]
        tempDf=tempDf[tempDf>=preThed]
        if tempDf.shape[0]==0:
            allSampleNum += 0
            allSampleList.append([])
            continue
        pList=tempDf.index.values
        # print(pList)
        # findSample(pList)

        sampleList = []
        pListInt = []
        for p in pList:
            pListInt.append(int(p))
        tempPoints = pointsDf.loc[pListInt, :]
        # print(tempPoints)
        clist = tempPoints.columns
        tempPoints['Dist'] = tempPoints.apply(lambda x: math.sqrt((x[clist[0]] - 0) ** 2 + (90 - x[clist[1]]) ** 2),
                                              axis=1)
        tempPoints.sort_values(by='Dist', ascending=True)
        rows, cols = tempPoints.shape[0], tempPoints.shape[1]

        sampleList.append(tempPoints.index[0])

        for j in range(1, rows):
            overlap=False
            # idx=tempPoints.index[i]
            # id = tempPoints.index.values.tolist().index(idx)
            idxlon, idxlat = tempPoints.iloc[j, 0], tempPoints.iloc[j, 1]

            for s in sampleList:
                ids = tempPoints.index.values.tolist().index(s)
                slon, slat = tempPoints.iloc[ids, 0], tempPoints.iloc[ids, 1]
                cover = (winWidth - abs(slon - idxlon)) * (winHeight - abs(slat - idxlat)) / (winWidth * winHeight)
                if cover > overlapThred:
                    overlap= True
                    break

            if overlap==False:
                sampleList.append(tempPoints.index[j])

        pListInt = []
        for s in sampleList:
            pListInt.append(int(s))
        tempOutDf = tempPoints.loc[pListInt, :]
        # print(tempOutDf)
        allSampleNum+=len(sampleList)
        allSampleList.append(sampleList)

        if i==20190101:
            print(len(sampleList), sampleList)
        # break
    print(allSampleNum, allSampleList)
    return 0
def main():
    func_3()
    return 'OK!'
main()