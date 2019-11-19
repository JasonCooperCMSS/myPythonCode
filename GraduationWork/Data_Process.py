# coding:utf-8
import os,shutil,time,math,random
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td

while(True):
    try:
        import arcpy
    except:
        continue
    else:
        break
from arcpy import env
arcpy.CheckOutExtension("spatial")
arcpy.env.overwriteOutput = True

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
# 3小时地面观测站数据，合并成日度数据，以11-08共8个数值相加，作为当天降水量
# 因为IMERG是UTC时。对每日缺失的不同处理，以maxMissing标定。
def func_1(inFile="", outFile="", maxMissing=7, btime=11, etime=8):  #
    inPath = "F:/Test/GraduationWork/Data/Temp"
    inFile = os.path.join(inPath, "SK_323_20190802.csv") #SK_323_201801_06
    outFile = os.path.join(inPath, "TempDaily.csv")
    inDf = pd.read_csv(inFile, header=0, index_col=0)

    nrows,ncols=inDf.shape[0],inDf.shape[1]
    # if nrows !=488 or ncols!=82:
    #     raise ValueError("Origin data was wrong !")
    index,names=inDf.index,inDf.columns
    outDf = pd.DataFrame()
    for j in range(0,ncols):
        numMiss, preSum = 0, 0
        newIndex ,dataList=[],[]
        for i in range(0,nrows):
            if inDf.iloc[i, j] == np.nan:
                numMiss += 1
            preSum += inDf.iloc[i, j]
            if index[i]%100==etime:
                date = (dt.strptime(str(index[i] / 100),"%Y%m%d")+td(days=-1)).strftime("%Y%m%d")
                newIndex.append(date)
                if numMiss<=maxMissing:
                    dataList.append(preSum)
                else:
                    dataList.append(np.nan)
                numMiss, preSum = 0, 0

        temmDf=pd.DataFrame(data=dataList,index=newIndex,columns=[names[j]])
        outDf=outDf.join(temmDf,how='right')
        print("col {} is done!".format(j))
        # break
    # outDf.columns=newIndex
    outDf.index.name = 'RGS'
    print(outDf.head(5))
    print(outDf.tail(5))
    outDf.to_csv(outFile)
    return 0
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
    tempDf = rgssDf.loc[20190101, :]
    # print(tempDf)
    tempDf = tempDf[tempDf >= preThed]
    pList = tempDf.index.values
    # print(pList)
    # findSample(pList)
    pListInt = []
    for p in pList:
        pListInt.append(int(p))
    tempPoints = pointsDf.loc[pListInt, :]
    print(tempPoints)
    "--------------------------------------------------"
    # def findSample(pList):
    #     numSample=0
    #     sampleList=[]
    #     pListInt=[]
    #     for p in pList:
    #         pListInt.append(int(p))
    #     tempPoints=pointsDf.loc[pListInt,:]
    #     print(tempPoints)
    #     clist=tempPoints.columns
    #     tempPoints['Dist']=tempPoints.apply(lambda x: math.sqrt((x[clist[0]]-0)**2+(90-x[clist[1]])**2), axis=1)
    #     tempPoints.sort_values(by='Dist',ascending=True)
    #     rows,cols=tempPoints.shape[0],tempPoints.shape[1]
    #
    #     sampleList.append(tempPoints.index[0])
    #     def isOverlap(idx,sampleList):
    #         id = tempPoints.index.values.tolist().index(idx)
    #         idxlon, idxlat = tempPoints.iloc[id, 0], tempPoints.iloc[id, 1]
    #
    #         for s in sampleList:
    #             ids=tempPoints.index.values.tolist().index(s)
    #             slon,slat=tempPoints.iloc[ids, 0],tempPoints.iloc[ids, 1]
    #
    #             overlap=(winWidth-abs(slon-idxlon))*(winHeight-abs(slat-idxlat))/(winWidth*winHeight)
    #             if overlap>overlapThred:
    #                 return False
    #             else:
    #                 continue
    #         return True
    #     for i in range(1,rows):
    #         if isOverlap(tempPoints.index[i],sampleList):
    #             sampleList.append(tempPoints.index[i])
    #     pListInt=[]
    #     for s in sampleList:
    #         pListInt.append(int(s))
    #     tempOutDf=tempPoints.loc[pListInt,:]
    #     print(tempOutDf)
    #     return numSample,sampleList
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

        # def isOverlap(idx, sampleList):
        #     id = tempPoints.index.values.tolist().index(idx)
        #     idxlon, idxlat = tempPoints.iloc[id, 0], tempPoints.iloc[id, 1]
        #
        #     for s in sampleList:
        #         ids = tempPoints.index.values.tolist().index(s)
        #         slon, slat = tempPoints.iloc[ids, 0], tempPoints.iloc[ids, 1]
        #
        #         overlap = (winWidth - abs(slon - idxlon)) * (winHeight - abs(slat - idxlat)) / (winWidth * winHeight)
        #         if overlap > overlapThred:
        #             return False
        #         else:
        #             continue
        #     return True

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

# QPE 3 h to 1 day
def func_4(inFile="", outFile="", maxMissing=7, btime=11, etime=8):  #
    inPath = "F:/Test/GraduationWork/Data/Temp/QPE/0505"
    inFile = os.path.join(inPath, "SK_323_20190802.csv") #SK_323_201801_06
    outFile = os.path.join(inPath, "TempDaily.csv")
    outRas = 'F:/Test/GraduationWork/Data/Temp/QPE/QPE_20190505.tif'
    env.workspace = "F:/Test/GraduationWork/Data/Temp/QPE"
    filesList=[]
    # i=0
    for f in os.listdir(inPath):
        file=os.path.join(inPath,f)
        try:
            tempRas = Raster(file)
        except:
            continue
        filesList.append(tempRas)
    outPlus=filesList[0]
    for i in range(1,len(filesList)):
        outPlus = Plus(outPlus, filesList[i])
    outPlus.save(outRas)

# 值提取到点,再合并到一个表
def func_5():
    inPath = "F:/Test/GraduationWork/Data/Temp/HBAL"
    inFile = os.path.join(inPath, "SK_323_20190802.csv")  # SK_323_201801_06
    outFile = os.path.join(inPath, "TempDaily.csv")
    points= "F:/Test/GraduationWork/Data/Points/Points82.shp"
    outCsv = 'F:/Test/GraduationWork/Data/Temp/PreOf82RGS.csv'
    env.workspace =inPath
    rasList=arcpy.ListRasters()
    env.workspace = "F:/Test/GraduationWork/Data/Temp/0/"
    for ras in rasList:

        namePieces=ras.split('_')
        tempShp=namePieces[0]+namePieces[1]+'_RG82.shp'
        file=os.path.join(inPath,ras)
        print(file)
        ExtractValuesToPoints(points, file, tempShp,"INTERPOLATE", "VALUE_ONLY")

        outExcels = namePieces[0]+namePieces[1]+'_RG82' + '.xls'
        arcpy.TableToExcel_conversion(tempShp, outExcels)  # "表转Excel"

# 计算相关系数
def staAna(x, y):
    x=np.array(x,dtype=np.float)
    y=np.array(y,dtype=np.float)

    xave = [sum(x) / len(x) for i in range(0, len(x))]
    yave = [sum(y) / len(y) for i in range(0, len(y))]
    xe = x - xave
    ye = y - yave

    R2 = sum(xe * ye) ** 2 / (sum(xe ** 2) * sum(ye ** 2))
    RMSE = np.sqrt(sum((y - x) ** 2) / len(x))
    Bias = sum(y) / sum(x) - 1
    MAE = sum(abs(y - x)) / len(x)

    r = []
    r.append(R2[0])
    r.append(RMSE[0])
    r.append(Bias[0])
    r.append(MAE[0])
    return r

def func_6():
    path="F:/Test/GraduationWork/Data/Temp/0/"
    infile=path+"StaDataOf4Days.xls"
    inDf=pd.read_excel(infile,header=0)
    rows,cols=inDf.shape[0],inDf.shape[1]
    for c in range(0,cols,3):
        rgs=inDf.iloc[:,c].values
        imerg=inDf.iloc[:,c+1].values
        qpe=inDf.iloc[:,c+2].values

        r1=staAna(rgs,imerg)
        r2=staAna(rgs,qpe)

        print(r1)
        print(r2)
        # break
    # print(inDf)

# IMERGD数据处理
def preProcess_IMERGD():
    inPath="F:/Test/GraduationWork/Data/IMERG/"
    ncPath=inPath+"Daily_nc_V06/"
    outTifPath=inPath+"Daily_tif_0.1/"
    outHBALPath = inPath + "Daily_HBAL/"
    outResampledPath=inPath+"Daily_HBAL_0.05/"
    workspace=inPath + "0/"
    mask="F:/Test/GraduationWork/Data/HB/"+"HubeiAreaLarge.shp"
    for f in os.listdir(ncPath):
        env.workspace=workspace
        ncfile=os.path.join(ncPath,f)
        date=f.split('-')[1].split('.')[-1]

        layer = 'nc_' + date
        arcpy.MakeNetCDFRasterLayer_md(ncfile, "precipitationCal", "lon", "lat", layer)  # "nc制作图层"

        outTif=outTifPath+"IMERGD_"+date+".tif"
        arcpy.Resample_management(layer, outTif, "0.1", "BILINEAR")

        outExtractByMask = arcpy.sa.ExtractByMask(outTif, mask)
        outHBAL=outHBALPath+"IMERGD_"+date+"_HBAL"+".tif"
        outExtractByMask.save(outHBAL)

        outTif = outResampledPath + "Resampled_IMERGD_" + date + "_HBAL"+".tif"
        arcpy.Resample_management(outHBAL, outTif, "0.05", "BILINEAR")
        print("{} is done!".format(date))
        # break

    return 0

# 把 OpGP每八小时合并成一天
def preProcess_OPGP():  #
    inPath = "F:/Test/GraduationWork/Data/OPGP/"
    hour3Path=inPath+"OPGP_3h/"
    dailyPath=inPath+"Daily_tif_0.05/"
    HBALPath=inPath+"Daily_HBAL_0.05/"

    rasSample=arcpy.Raster("F:/Test/GraduationWork/Data/IMERG/"+"Daily_tif_0.1/IMERGD_20190630.tif")
    sr=rasSample.spatialReference
    mask="F:/Test/GraduationWork/Data/HB/"+"HubeiAreaLarge.shp"

    env.workspace =inPath+"0/"
    filesList=[]
    # i=0
    for f in os.listdir(hour3Path):
        time=f.split('.')[0].split('-')[-1]
        hour=time[-2:]

        file=os.path.join(hour3Path,f)
        try:
            tempRas = arcpy.Raster(file)
        except:
            continue
        filesList.append(tempRas)
        print(f)

        if hour =='08':
            # filesList.append(tempRas)
            # print(f)
            date = (dt.strptime((time[0:8]),"%Y%m%d")+td(days=-1)).strftime("%Y%m%d")
            outPlus = filesList[0]
            for i in range(1, len(filesList)):
                outPlus = arcpy.sa.Plus(outPlus, filesList[i])
                # Set environmental variables for output
            outTif=dailyPath+"OPGPD_"+date+".tif"
            arcpy.ProjectRaster_management(outPlus, outTif, sr, "BILINEAR")

            outExtractByMask = arcpy.sa.ExtractByMask(outTif, mask)
            outHBAL=HBALPath+"OPGPD_"+date+"_HBAL.tif"
            outExtractByMask.save(outHBAL)
            print ("date {} is combined by {} files!".format(date,len(filesList)))
            filesList=[]
            # filesList.append(tempRas)
            # break
"----------------------------------"
def calSSIM(rarr=0,larr=0):
    # 存在问题；
    # 1）在降水图像上C1/C2/C3值显得太大了，不能以图像领域中K1/K2/L=255计算.
    # 2) 为防止假象，SSIM一般要求应用在局部，对全图分块处理，甚至高斯加权华东平均得到 MSSIM才比较可行
    xave, yave = np.mean(rarr), np.mean(larr)
    xvar, yvar = np.var(rarr, ddof=1), np.var(larr, ddof=1)
    xstd, ystd = np.std(rarr, ddof=1), np.std(larr, ddof=1)
    xycov = np.cov(rarr, larr, rowvar=True, ddof=1)[0, 1]
    K1, K2, L = 0.01, 0.03, 50
    # C1,C2,C3=(K1*L)**2,(K2*L)**2,(K2*L)**2/2
    C1, C2, C3 = 0, 0, 0
    print(xave, yave, xvar, yvar, xstd, ystd)

    lxy = (2 * xave * yave + C1) / (xave ** 2 + yave ** 2 + C1)
    cxy = (2 * xstd * ystd + C2) / (xvar + yvar + C2)
    sxy = (xycov + C3) / (xstd * ystd + C3)
    print(lxy, cxy, sxy)
    SSIM = lxy * cxy * sxy
    print (SSIM)
    return 0
# preSampleNumberEstimate  预先样本数分析 全湖北省相似度评价
def preSamNumEstimate():
    leftPath="F:/Test/GraduationWork/Data/IMERG/Daily_HBAL_0.05/"
    rightPath = "F:/Test/GraduationWork/Data/OPGP/Daily_HBAL_0.05/"
    outPccCsv="F:/Test/GraduationWork/Data/Temp/SampleNumber/"+"DateWithPCC.csv"
    outAsdPccCsv = "F:/Test/GraduationWork/Data/Temp/SampleNumber/" + "DateSortByAsdPCC.csv"

    corrList,dateList=[],[]
    arcpy.env.workspace = rightPath
    for rf in arcpy.ListRasters('*','TIF'):
        date=rf.split('_')[1]
        opgpdFile=os.path.join(rightPath,rf)
        lf="Resampled_IMERGD_"+date+"_HBAL.tif"
        imergdFile=os.path.join(leftPath,lf)

        rarr=arcpy.RasterToNumPyArray(opgpdFile,nodata_to_value=0)
        larr=arcpy.RasterToNumPyArray(imergdFile,nodata_to_value=0)
        # print(rarr,larr)

        rarr = np.reshape(rarr, (1, -1))
        larr = np.reshape(larr, (1, -1))
        corr = np.corrcoef(rarr, larr, rowvar=True)
        print("{} PCC is {}. ".format(date,corr[0,1]))
        corrList.append(corr[0,1])
        dateList.append(date)
        # break

    pccDf=pd.DataFrame(data=corrList,index=dateList,columns=['PCC'])
    pccDf.to_csv(outPccCsv)
    PccDf=pccDf.sort_values(by='PCC',ascending=False)
    PccDf.to_csv(outAsdPccCsv)
    return 0

# 按得到的顺序，把OPGP数据另复制一份，命名前加顺序，以00001开始
def func_7():
    inPath="F:/Test/GraduationWork/Data/IMERG/Daily_HBAL_0.05/"
    outPath="F:/Test/GraduationWork/Data/IMERG/SortAsd_HBAL_0.05/"
    indexCsv="F:/Test/GraduationWork/Data/Temp/SampleNumber/" + "DateSortByAsdPCC.csv"

    df=pd.read_csv(indexCsv,header=0,index_col=0)
    i=1
    for index in df.index:
        print(i)
        date=str(index)
        fin="Resampled_IMERGD_"+date+"_HBAL.tif"
        infile=os.path.join(inPath,fin)
        fout=str(10000+i)[-4:]+"_Resampled_IMERGD_"+date+"_HBAL.tif"
        outfile=os.path.join(outPath,fout)
        arcpy.Copy_management(in_data=infile, out_data=outfile)
        i+=1

"dwwwwwwwwwwwww"
def calSSIM(rarr=0,larr=0):
    # 存在问题；
    # 1）在降水图像上C1/C2/C3值显得太大了，不能以图像领域中K1/K2/L=255计算.
    # 2) 为防止假象，SSIM一般要求应用在局部，对全图分块处理，甚至高斯加权华东平均得到 MSSIM才比较可行
    rarr,larr=rarr.astype(np.float64),larr.astype(np.float64)
    xave, yave = np.mean(rarr), np.mean(larr)
    xvar, yvar = np.var(rarr, ddof=1), np.var(larr, ddof=1)
    xstd, ystd = np.std(rarr, ddof=1), np.std(larr, ddof=1)
    xycov = np.cov(rarr, larr, rowvar=True, ddof=1)[0, 1]
    K1, K2, L = 0.01, 0.03, max(np.max(rarr),np.max(larr))/2
    C1,C2,C3=(K1*L)**2,(K2*L)**2,(K2*L)**2/2
    # C1, C2, C3 = 0, 0, 0
    # print(xave, yave, xvar, yvar, xstd, ystd)

    lxy = (2 * xave * yave + C1) / (xave ** 2 + yave ** 2 + C1)
    cxy = (2 * xstd * ystd + C2) / (xvar + yvar + C2)
    sxy = (xycov + C3) / (xstd * ystd + C3)
    # print(lxy, cxy, sxy)
    SSIM = lxy * cxy * sxy
    # print (SSIM)
    return SSIM
# 每对图像的优选
def winSelection(leftFile="", rightFile="", widthWin=40, heightWin=40, step=10, minNoRainRatio=0.2, corrThred=0.5,
                 overlapThred=0.5):
    # leftFile = 'F:/Test/GraduationWork/Data/IMERG/SortAsd_HBAL_0.05/'+'0062_Resampled_IMERGD_20190621_HBAL.tif'
    # rightFile ='F:/Test/GraduationWork/Data/OPGP/SortAsd_HBAL_0.05/'+'0062_OPGPD_20190621_HBAL.tif'
    idDate,date=rightFile.split('/')[-1].split('_')[0],rightFile.split('/')[-1].split('_')[-2]
    winListFile="F:/Test/GraduationWork/Data/Temp/SampleNumber/temp_1/"+idDate+'_'+date+"_winInfo.csv" #DailySampleInfo

    sampleRas=arcpy.Raster(rightFile)
    xUpLeft,yUpLeft=sampleRas.extent.XMin,sampleRas.extent.YMax
    cellWidth,cellHeight=sampleRas.meanCellWidth,sampleRas.meanCellHeight

    larr = arcpy.RasterToNumPyArray(leftFile, nodata_to_value=0)
    rarr = arcpy.RasterToNumPyArray(rightFile, nodata_to_value=0)
    winDf=pd.DataFrame(columns=['row','col','corr'])
    rows,cols=rarr.shape[0],rarr.shape[1]

    for r in range(0,rows-heightWin+1,step):
        for c in range(0,cols-widthWin+1,step):
            rwinArr=rarr[r:r+heightWin,c:c+widthWin]
            # print(tempRarr)
            noRainRatio=len(rwinArr[rwinArr<0.1])/float(widthWin*heightWin)
            if noRainRatio>minNoRainRatio:
                continue
            else:
                lwinArr = larr[r:r+heightWin,c:c+widthWin]
                # tempCorr=np.corrcoef(rwinArr.reshape(1,-1),lwinArr.reshape(1,-1),rowvar=0,ddof=1)[0,1]
                tempCorr=calSSIM(rwinArr.reshape(1,-1),lwinArr.reshape(1,-1))
                "-----------------------------"
                if tempCorr<corrThred:
                    continue
                else:
                    tempDf = pd.DataFrame(data={'row':[r], 'col':[c], 'corr':[tempCorr]})
                    winDf=pd.concat((winDf,tempDf),axis=0,ignore_index=True)
                    # print(tempCorr)
                    # break
        # break
    if winDf.shape[0]==0:
        return 0
    winDf=winDf.sort_values(by='corr',ascending=False)
    winDf = winDf.reindex(columns=['row', 'col', 'corr'])
    winDf.index= [i for i in range(winDf.shape[0])]

    # print(winDf.tail())

    outWinDf = pd.DataFrame(data={'row': [winDf.iloc[0, 0]], 'col': [winDf.iloc[0, 1]], 'corr': [winDf.iloc[0, 2]]},
                            columns=['row', 'col', 'corr'])
    # print(outWinDf)
    for idx in range(1, winDf.shape[0]):
        overlap = False
        xCenterIdx, yCenterIdx = winDf.iloc[idx, 0] + widthWin / 2, winDf.iloc[idx, 1] + heightWin / 2
        for s in outWinDf.index:
            # print(s)
            xCenterS, yCenterS = outWinDf.loc[s, 'row'] + widthWin / 2, outWinDf.loc[s, 'col'] + heightWin / 2
            cover = (widthWin - abs(xCenterS - xCenterIdx)) * (heightWin - abs(yCenterS - yCenterIdx)) / (
                        widthWin * heightWin)
            if cover > overlapThred:
                overlap = True
                break
        if overlap == False:
            tempDf = pd.DataFrame(
                data={'row': [winDf.iloc[idx, 0]], 'col': [winDf.iloc[idx, 1]], 'corr': [winDf.iloc[idx, 2]]},
                columns=['row', 'col', 'corr'])
            outWinDf = pd.concat((outWinDf, tempDf), axis=0, ignore_index=True)
            # print(outWinDf)
    print(outWinDf)
    clist=['row', 'col', 'corr','lonWinCenter','latWinCenter']
    outWinDf=outWinDf.reindex(columns=clist)
    outWinDf['lonWinCenter'] = outWinDf.apply(lambda x: xUpLeft + (x['col'] + widthWin / 2) * cellWidth, axis=1)
    outWinDf['latWinCenter'] = outWinDf.apply(lambda x: yUpLeft - (x['row'] + heightWin / 2) * cellWidth, axis=1)
    outWinDf.to_csv(winListFile)
    return outWinDf.shape[0]
# 临时试探代码

def func_8():
    leftPath = "F:/Test/GraduationWork/Data/IMERG/SortAsd_HBAL_0.05/"
    rightPath="F:/Test/GraduationWork/Data/OPGP/SortAsd_HBAL_0.05/"
    arcpy.env.workspace=rightPath
    count=0
    sampleCount=0
    for f in arcpy.ListRasters("*","TIF"):
        id, date = f.split("_")[0], f.split("_")[2]
        rf = os.path.join(rightPath, f)
        lf = os.path.join(leftPath, id + "_Resampled_IMERGD_" + date + "_HBAL.tif")
        tempCount = winSelection(leftFile=lf, rightFile=rf, widthWin=60, heightWin=60, step=5, minNoRainRatio=0.05,
                                 corrThred=0, overlapThred=0.25)
        sampleCount+=tempCount
        print("{} has {} window-pair!".format(date,tempCount))
        # break
        if count>10:
            break
        count += 1
    print(sampleCount)

def func_9():

    widthWin,heightWin,cellsize=40,40,0.05
    originalPath="F:/Test/GraduationWork/Data/IMERG/Daily_HBAL/"
    leftPath="F:/Test/GraduationWork/Data/IMERG/SortAsd_HBAL_0.05/"
    rightPath="F:/Test/GraduationWork/Data/OPGP/SortAsd_HBAL_0.05/"

    # winListPath="F:/Test/GraduationWork/Data/Temp/SampleNumber/"+"setup_1/"
    winListPath = "F:/Test/GraduationWork/Data/Temp/SampleNumber/" + "temp_1"  #DailySampleInfo
    outPath = "F:/Test/GraduationWork/Data/Temp/tempExtract/"+"DailySampleInfo/"        #tempExtract_setup_1
    if os.path.exists(outPath)==False:
        os.mkdir(outPath)

    count=1
    for f in os.listdir(winListPath):
        idf,date=f.split('_')[0],f.split('_')[1]

        winListFile=os.path.join(winListPath,f)
        originalFile=originalPath+"IMERGD_"+date+"_HBAL.tif"
        leftFile=leftPath+idf+"_Resampled_IMERGD_"+date+"_HBAL.tif"
        rightFile=rightPath+idf+"_OPGPD_"+date+"_HBAL.tif"
        # outPath="F:/Test/GraduationWork/Data/Temp/tempExtract/"

        oRas=arcpy.Raster(originalFile)
        rRas=arcpy.Raster(rightFile)
        lRas=arcpy.Raster(leftFile)
        arcpy.env.overwriteOutput = True
        arcpy.env.outputCoordinateSystem = rRas

        oarr=arcpy.RasterToNumPyArray(oRas,nodata_to_value=0)
        rarr=arcpy.RasterToNumPyArray(rRas,nodata_to_value=0)
        larr=arcpy.RasterToNumPyArray(lRas,nodata_to_value=0)

        winDf=pd.read_csv(winListFile,header=0,index_col=0)
        for r in range(winDf.shape[0]):
            lowerLeft=arcpy.Point(rRas.extent.XMin+winDf.iloc[r,1]*cellsize,rRas.extent.YMin+(rRas.height-heightWin-winDf.iloc[r,0])*cellsize)

            tempArr = rarr[winDf.iloc[r, 0]:winDf.iloc[r, 0] + widthWin, winDf.iloc[r, 1]:winDf.iloc[r, 1] + heightWin]
            tempRaster = arcpy.NumPyArrayToRaster(tempArr, lowerLeft, rRas.meanCellWidth, rRas.meanCellHeight)
            tempRaster.save(outPath+idf+'_OPGPD_'+date+'_HBAL_P'+str(r)+".tif")

            tempArr = larr[winDf.iloc[r, 0]:winDf.iloc[r, 0] + widthWin, winDf.iloc[r, 1]:winDf.iloc[r, 1] + heightWin]
            tempRaster = arcpy.NumPyArrayToRaster(tempArr, lowerLeft, rRas.meanCellWidth, rRas.meanCellHeight)
            tempRaster.save(outPath + idf + '_Resampled_IMERGD_' + date + '_HBAL_P' + str(r) + ".tif")

            tempArr = oarr[winDf.iloc[r, 0]/2:(winDf.iloc[r, 0] + widthWin)/2, winDf.iloc[r, 1]/2:(winDf.iloc[r, 1] + heightWin)/2]
            tempRaster = arcpy.NumPyArrayToRaster(tempArr, lowerLeft, rRas.meanCellWidth*2, rRas.meanCellHeight*2)
            tempRaster.save(outPath+idf+'_Original_IMERGD_'+date+'_HBAL_P' + str(r) + ".tif")
        # break
        if count>10:
            break
        count += 1
    return 0
def main():
    # temp()
    func_1()
    # func_2(numToDel=50)
    # func_3()
    return 'OK!'
main()