# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
import sys, string, os
import xlrd
import xlwt
import xlsxwriter
import numpy as np
import time
import math
import random
import sympy
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import random


#   批量nc转tif
def f1():
    pathIn = 'F:/Test/Data/IMERG/' + 'IMERGM_nc_201501_201805_HB91_51/'
    pathOut = 'F:/Test/Data/IMERG/' + 'IMERGM_tif_201501_201805_HB91_51/'
    arcpy.env.workspace = pathIn
    for nc_file in arcpy.ListFiles("*.nc"):
        time = nc_file[20:28]
        y, m, d = int(time[0:4]), int(time[4:6]), int(time[6:8])
        layer = 'nc_' + time
        arcpy.MakeNetCDFRasterLayer_md(nc_file, "precipitation", "lon", "lat", layer)  # "nc制作图层"
        if (m == 2):
            if (y == 2016):
                times = 29 * 24
            else:
                times = 28 * 24
        elif (m == 4 or m == 6 or m == 9 or m == 11):
            times = 30 * 24
        else:
            times = 31 * 24
        print y, m, d
        outTimes = Times(layer, times)
        outTimes.save(pathOut + 'I' + time[0:6] + '.tif')


#   GCS下批量重采样到0.1度，再按掩膜提取到96*96和80*48，再把原始IMERGM降水值提取到83个站点。
def f2_GCS():
    path = 'F:/Test/Paper180829/Data/IMERG/'
    arcpy.env.workspace = path + 'originIMERG/'
    rasters = arcpy.ListRasters()
    for raster in rasters:
        outResample = path + '0.1dIMERG/' + raster
        arcpy.Resample_management(raster, outResample, "0.1", "BILINEAR")  # "重采样到0.1度"

        mask = 'F:/Test/Paper180829/Data/HB/' + 'HB8048.shp'
        outExtract = ExtractByMask(outResample, mask)  # "按掩膜提取"

        outExtract.save(path + '0.1d8048/' + raster)

        points = 'F:/Test/Paper180829/Data/POINT/' + 'POINTS83.shp'
        outPoints = path + '0.1dRG83/' + raster[0:7] + '.shp'
        ExtractValuesToPoints(points, outExtract, outPoints, "INTERPOLATE", "VALUE_ONLY")  # "值提取到点"

        outExcels = path + '0.1dExcels/' + raster[0:7] + '.xls'
        arcpy.TableToExcel_conversion(outPoints, outExcels)  # "表转Excel"


#   GCS下原始0.1度数据值提取到3840点存起来，以便分析原始数据精度
def f22():
    path = 'F:/Test/Paper180829/data/IMERG/'
    arcpy.env.workspace = path + 'originIMERG/'
    rasters = arcpy.ListRasters()
    for raster in rasters:

        outExtract = path + '0.1d8048/' + raster

        points = 'F:/Test/Paper180829/Data/POINT/' + 'POINTS3840.shp'
        outPoints = path + '0.1dRG3840/' + raster[0:7] + '.shp'
        ExtractValuesToPoints(points, outExtract, outPoints, "INTERPOLATE", "VALUE_ONLY")  # "值提取到点"

        arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))

        outExcels = path + '0.1d_Excels3840/' + raster[0:7] + '.xlsx'
        workbook = xlsxwriter.Workbook(outExcels)
        worksheet = workbook.add_worksheet('0.1d_IMERG')  # originIMERG 0.1dIMERG
        # print len(arr),arr[0],arr[0][0]
        for i in range(0, len(arr)):
            worksheet.write(i, 0, arr[i][0])
        workbook.close()


#   先把originIMERG投影到UTM，在UTM下批量重采样到10km，再按掩膜提取到96*96和80*48，再把原始IMERGM降水值提取到83个站点。
def f2_UTM():
    path = 'F:/Test/Paper180829/Data/IMERG/'
    arcpy.env.workspace = path + 'originIMERG/'
    rasters = arcpy.ListRasters()
    for raster in rasters:
        outUTM = path + 'UTMIMERG/' + raster
        arcpy.ProjectRaster_management(raster, outUTM, \
                                       "32649", "BILINEAR")  # "WGS_1984_UTM_Zone_49N.prj","BILINEAR"

        outResample = path + '10kmIMERG/' + raster
        arcpy.Resample_management(outUTM, outResample, "10000", "BILINEAR")  # "重采样到10km"

        mask = 'F:/Test/Paper180829/Data/HB/' + 'UTMHB8048.shp'
        outExtract = ExtractByMask(outResample, mask)  # "按掩膜提取"
        outExtract.save(path + '10km8048/' + raster)

        points = 'F:/Test/Paper180829/Data/POINT/' + 'UTMPOINTS83.shp'
        outPoints = path + '10kmRG83/' + raster[0:7] + '.shp'
        ExtractValuesToPoints(points, outExtract, outPoints, "INTERPOLATE", "VALUE_ONLY")  # "值提取到点"

        outExcels = path + '10kmExcels/' + raster[0:7] + '.xls'
        arcpy.TableToExcel_conversion(outPoints, outExcels)  # "表转Excel"


#   把提取到的IMERG降水值，汇总到一张表中
def f3():
    # path='F:/Test/Paper180829/Data/IMERG/'
    path = 'F:/Test/Paper180829/process/UTMMF/'
    filesPath = path + 'Excels/'  # originExcels 0.1dExcels   10kmExcels
    files = os.listdir(filesPath)
    inputFiles = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.xls':  # "文件获取"
            inputFiles.append(filesPath + files[i])

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('UTMMF8048')  # originIMERG 0.1dIMERG 10kmIMERG
    c = 0  # 列变量
    for i in range(0, len(inputFiles)):  # len(inputFiles)
        excel = xlrd.open_workbook(inputFiles[i])
        table = excel.sheet_by_index(0)
        rows = table.nrows
        r = 1  # 行变量,取1是为了留出列标题月份
        for j in range(1, rows):  # 注意跳过第一行的标题行
            temp = table.cell(j, 3).value  # 注意像元值RASTERVALU所在的列数
            worksheet.write(j, c, temp)
        c = c + 1

    outFile = path + "UTMMFoutFile.xls"  # originIMERG.xls  0.1dIMERG.xls    10kmIMERG
    workbook.save(outFile)


def f4(x, y):
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


#   GCS计算精度指标
def f5():
    path = 'F:/Test/Paper180829/Results/'
    data = xlrd.open_workbook(path + 'Validation.xls')
    t1 = data.sheet_by_index(0)
    t2 = data.sheet_by_index(4)  # 1  2  3取第几个表

    wb = xlwt.Workbook(encoding='utf-8')
    ws1 = wb.add_sheet('UTMMF8048')  # originIMERG  0.1DIMERG   10kmIMERG
    #   控制计算范围
    for i in range(0, 36):
        c1 = t1.col_values(i, 1, 84)
        c2 = t2.col_values(i, 1, 84)
        #   全部转浮点，以防万一
        for j in range(0, 83):
            c1[j] = float(c1[j])
            c2[j] = float(c2[j])
        RGS = np.array(c1)
        IMERG = np.array(c2)
        r1 = f4(RGS, IMERG)  # 注意变量传入位置，应是RGS在前
        for j in range(0, 4):
            ws1.write(j, i, r1[j])
    out = path + "UTMMF8048.xls"  # originIMERGSTA.xls 0.1DIMERGSTA.xls
    wb.save(out)


#   批量计算三年月平均降水量
def f6():
    path = 'F:/Test/Paper180829/Data/IMERG/'
    filesPath = path + '10km8048/'  # 0.1d8048/
    files = os.listdir(filesPath)
    ras = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.tif':  # "文件获取"
            ras.append(filesPath + files[i])
    for i in range(0, 12):
        outRas = (Raster(ras[i]) + Raster(ras[i + 12]) + Raster(ras[i + 24])) / 3
        outRas.save(path + '3Mean_UTMHB8048/' + str(i + 1) + '.tif')  # 3Mean_HB8048


#   对LST进行合并到月，关键是对nodata值的处理
def f7():
    for y in range(2015, 2018):
        path = 'F:/Test/Data/LST/'
        path2 = path + 'LTN8D/' + str(y) + '/'  # LTD8D
        files = os.listdir(path2)
        Input_file = []
        for i in range(0, len(files)):
            if os.path.splitext(files[i])[1] == '.tif':  # "文件获取"
                Input_file.append(path2 + files[i])

        mFiles = [[] for i in range(0, 12)]
        for i in range(0, len(Input_file)):
            if (int(Input_file[i][41:44]) > 0 and int(Input_file[i][41:44]) < 31):
                mFiles[0].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 31 and int(Input_file[i][41:44]) < 53):
                mFiles[1].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 53 and int(Input_file[i][41:44]) < 60):
                mFiles[1].append(Input_file[i])
                mFiles[2].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 60 and int(Input_file[i][41:44]) < 83):
                mFiles[2].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 83 and int(Input_file[i][41:44]) < 91):
                mFiles[2].append(Input_file[i])
                mFiles[3].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 91 and int(Input_file[i][41:44]) < 121):
                mFiles[3].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 121 and int(Input_file[i][41:44]) < 152):
                mFiles[4].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 152 and int(Input_file[i][41:44]) < 174):
                mFiles[5].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 174 and int(Input_file[i][41:44]) < 182):
                mFiles[5].append(Input_file[i])
                mFiles[6].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 182 and int(Input_file[i][41:44]) < 206):
                mFiles[6].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 206 and int(Input_file[i][41:44]) < 213):
                mFiles[6].append(Input_file[i])
                mFiles[7].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 213 and int(Input_file[i][41:44]) < 236):
                mFiles[7].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 236 and int(Input_file[i][41:44]) < 244):
                mFiles[7].append(Input_file[i])
                mFiles[8].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 244 and int(Input_file[i][41:44]) < 268):
                mFiles[8].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 268 and int(Input_file[i][41:44]) < 274):
                mFiles[9].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 274 and int(Input_file[i][41:44]) < 297):
                mFiles[9].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 297 and int(Input_file[i][41:44]) < 305):
                mFiles[9].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 305 and int(Input_file[i][41:44]) < 327):
                mFiles[10].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 327 and int(Input_file[i][41:44]) < 335):
                mFiles[10].append(Input_file[i])
                mFiles[11].append(Input_file[i])
            elif (int(Input_file[i][41:44]) >= 335 and int(Input_file[i][41:44]) < 3666):
                mFiles[11].append(Input_file[i])
        # for i in range(0, 12):
        #     print len(mFiles[i])
        for m in range(0, 12):
            ras = Raster(mFiles[m][0])
            arcpy.env.overwriteOutput = True
            arcpy.env.outputCoordinateSystem = ras

            outWidth = ras.meanCellWidth
            outHeight = ras.meanCellHeight
            lowerLeft = arcpy.Point(ras.extent.XMin, ras.extent.YMin)

            arrTemp = arcpy.RasterToNumPyArray(ras)

            rows = arrTemp.shape[0]
            cols = arrTemp.shape[1]
            # print rows,'\n',cols
            arr = np.zeros((len(mFiles[m]), rows, cols), np.float)
            for j in range(0, len(mFiles[m])):
                temp = Raster(mFiles[m][j])
                arr[j] = arcpy.RasterToNumPyArray(temp)

            arrSum = np.zeros((rows, cols), np.float)

            for r in range(0, rows):
                for c in range(0, cols):
                    counts = 0
                    for j in range(0, len(mFiles[m])):
                        if (arr[j, r, c] >= 7500):
                            arrSum[r, c] = arrSum[r, c] + arr[j, r, c]
                            counts = counts + 1
                    if (counts != 0):
                        arrSum[r, c] = arrSum[r, c] / counts

            for i in range(0, rows):
                for j in range(0, cols):
                    while (arrSum[i, j] == 0):
                        direction = random.randint(1, 4)
                        if direction == 1:
                            if i - 1 > 0:
                                arrSum[i, j] = arrSum[i - 1, j]
                            else:
                                arrSum[i, j] = arrSum[i, j]
                        elif direction == 2:
                            if j + 1 < cols:
                                arrSum[i, j] = arrSum[i, j + 1]
                            else:
                                arrSum[i, j] = arrSum[i, j]
                        elif direction == 3:
                            if i + 1 < rows:
                                arrSum[i, j] = arrSum[i + 1, j]
                            else:
                                arrSum[i, j] = arrSum[i, j]
                        else:
                            if j - 1 > 0:
                                arrSum[i, j] = arrSum[i, j - 1]
                            else:
                                arrSum[i, j] = arrSum[i, j]

            tempRaster = arcpy.NumPyArrayToRaster(arrSum * 0.02 - 273.15, lowerLeft, outWidth, outHeight, -273.15)
            tempRaster.save(
                'F:/Test/Paper180829/Data/LTN/monthlyLTN/' + 'LTN' + str(y * 100 + m + 1) + '.tif')  # monthlyLTD

    print '啦啦啦'


#   对LST批量处理，值提取到点。
def f8_GCS():
    path = 'F:/Test/Paper180829/Data/LTN/'
    arcpy.env.workspace = path + 'monthlyLTN/'
    rasters = arcpy.ListRasters()
    for raster in rasters:
        # outResample=path+'0.0125d_rawLTN/'+raster
        # arcpy.Resample_management(raster, outResample, "0.0125", "BILINEAR")   #"重采样到0.1度"
        #
        # mask='F:/Test/Paper180829/Data/HB/'+'HB8048.shp'
        # outExtract = ExtractByMask(outResample, mask)   #"按掩膜提取"
        # outExtract.save(path+'0.0125d_LTN_HB8048/'+raster)
        #
        # outAggreg = Aggregate(outExtract, 8, "MEAN", "EXPAND", "DATA")
        # outAggreg.save(path+'0.1d_LTN_HB8048/'+raster)
        #
        # points='F:/Test/Paper180829/Data/POINT/'+'POINTS3840.shp'
        # outPoints=path + '0.1d_LTN_RG3840/' +raster[0:9]+'.shp'
        # ExtractValuesToPoints(points, outExtract, outPoints, "INTERPOLATE", "VALUE_ONLY")  #"值提取到点"
        #
        # arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))
        # # np.savetxt(path + '0.1d_LTN_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')
        # # print arr
        #
        # outExcels = path + '0.1d_LTN_Excels/' + raster[0:9] + '.xlsx'
        # workbook = xlsxwriter.Workbook(outExcels)
        # worksheet = workbook.add_worksheet('0.1d_LTN')  # originIMERG 0.1dIMERG
        # # print len(arr),arr[0],arr[0][0]
        # for i in range(0, len(arr)):
        #     worksheet.write(i, 0, arr[i][0])
        # workbook.close()

        outResample = path + '0.01d_rawLTN/' + raster
        arcpy.Resample_management(raster, outResample, "0.01", "BILINEAR")  # "重采样到0.1度"

        mask = 'F:/Test/Paper180829/Data/HB/' + 'HB8048.shp'
        outExtract = ExtractByMask(outResample, mask)  # "按掩膜提取"
        outExtract.save(path + '0.01d_LTN_HB8048/' + raster)

        points = 'F:/Test/Paper180829/Data/POINT/' + 'POINTS384000.shp'
        outPoints = path + '0.01d_LTN_RG384000/' + raster[0:9] + '.shp'
        ExtractValuesToPoints(points, outExtract, outPoints, "INTERPOLATE", "VALUE_ONLY")  # "值提取到点"

        arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))
        # np.savetxt(path + '0.01d_LTN_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')
        # print arr
        outExcels = path + '0.01d_LTN_Excels/' + raster[0:9] + '.xlsx'
        workbook = xlsxwriter.Workbook(outExcels)
        worksheet = workbook.add_worksheet('0.01d_LTD')  # originIMERG 0.1dIMERG
        # print len(arr),arr[0],arr[0][0]
        for i in range(0, len(arr)):
            worksheet.write(i, 0, arr[i][0])
        workbook.close()


#   UTM批量重采样到0.1度，再按掩膜提取到96*96和80*50，再把原始IMERGM降水值提取到83个站点。
def f8_UTM():
    path = 'F:/Test/Paper180829/Data/LTN/'
    arcpy.env.workspace = path + 'monthlyLTN/'
    rasters = arcpy.ListRasters()
    for raster in rasters:

        # maskIMERG = 'F:/Test/Paper180829/Data/HB/' + 'rangeOfOriginIMERG.shp'
        # outSubset= ExtractByMask(raster, maskIMERG)  # "按掩膜提取"
        # outSubset.save(path + 'subsetLTN/' + raster)
        #
        # outUTM=path+'rawLTNUTM/'+raster
        # arcpy.ProjectRaster_management(outSubset, outUTM, \
        #                        "32649", "BILINEAR") #"WGS_1984_UTM_Zone_49N.prj","BILINEAR"

        st = time.clock()
        outUTM = path + 'rawLTNUTM/' + raster
        outResample = path + '1km_rawUTMLTN/' + raster
        arcpy.Resample_management(outUTM, outResample, "1000", "BILINEAR")  # "重采样到10km"

        mask = 'F:/Test/Paper180829/Data/HB/' + 'UTMHB8048.shp'
        outExtract = ExtractByMask(outResample, mask)  # "按掩膜提取"
        outExtract.save(path + '1km_UTMLTN_HB8048/' + raster)

        # points='F:/Test/Paper180829/Data/POINT/'+'UTMPOINTS83.shp'
        points = 'F:/Test/Paper180829/Data/POINT/' + 'UTMPOINTS384000.shp'
        outPoints = path + '1km_UTMLTN_RG384000/' + raster[0:9] + '.shp'
        ExtractValuesToPoints(points, outExtract, outPoints, "INTERPOLATE", "VALUE_ONLY")  # "值提取到点"

        arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))
        # np.savetxt(path + '1km_UTMLTN_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')

        outExcels = path + '1km_UTMLTN_Excels/' + raster[0:9] + '.xlsx'
        workbook = xlsxwriter.Workbook(outExcels)
        worksheet = workbook.add_worksheet('1km_LTN')  # originIMERG 0.1dIMERG
        # print len(arr),arr[0],arr[0][0]
        for i in range(0, len(arr)):
            worksheet.write(i, 0, arr[i][0])
        workbook.close()
        print "所花时间为：", time.clock() - st, "秒"

        # outUTM=path+'rawLTNUTM/'+raster
        # outResample=path+'1.25km_rawUTMLTN/'+raster
        # arcpy.Resample_management(outUTM, outResample, "1250", "BILINEAR")   #"重采样到1.25km"
        #
        # mask='F:/Test/Paper180829/Data/HB/'+'UTMHB8048.shp'
        # outExtract = ExtractByMask(outResample, mask)   #"按掩膜提取"
        # outExtract.save(path+'1.25km_UTMLTN_HB8048/'+raster)
        #
        # outAggreg = Aggregate(outExtract, 8, "MEAN", "EXPAND", "DATA")    #"聚合到10km"
        # outAggreg.save(path+'10km_UTMLTN_HB8048/'+raster)
        #
        # points='F:/Test/Paper180829/Data/POINT/'+'UTMPOINTS3840.shp'
        # # points = 'F:/Test/Paper180829/Data/POINT/' + 'UTMPOINTS384000.shp'
        # outPoints=path + '10km_UTMLTN_RG3840/' +raster[0:9]+'.shp'
        # ExtractValuesToPoints(points, outAggreg, outPoints, "INTERPOLATE", "VALUE_ONLY")  #"值提取到点"
        #
        # arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))    #"将值转到xlsx"
        # # np.savetxt(path + '10km_UTMLTD_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')
        # outExcels = path + '10km_UTMLTN_Excels/' + raster[0:9] + '.xlsx'
        # workbook = xlsxwriter.Workbook(outExcels)
        # worksheet = workbook.add_worksheet('10km_LTN')  # originIMERG 0.1dIMERG
        # # print len(arr),arr[0],arr[0][0]
        # for i in range(0, len(arr)):
        #     worksheet.write(i, 0, arr[i][0])
        # workbook.close()


#   将DEM值转到xlsx
def f9():
    path = 'F:/Test/Paper180829/Data/DEM/'

    outPoints = path + 'DEMRG3840.shp'
    arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))
    # np.savetxt(path + '1km_UTMLTN_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')
    outExcels = path + '0.1dDEM' + '.xlsx'
    workbook = xlsxwriter.Workbook(outExcels)
    worksheet = workbook.add_worksheet('0.1dDEM')
    # print len(arr),arr[0],arr[0][0]
    for i in range(0, len(arr)):
        worksheet.write(i, 0, arr[i][0])
    workbook.close()

    outPoints = path + 'DEMRG384000.shp'
    arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))
    # np.savetxt(path + '1km_UTMLTN_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')
    outExcels = path + '0.01dDEM' + '.xlsx'
    workbook = xlsxwriter.Workbook(outExcels)
    worksheet = workbook.add_worksheet('0.01dDEM')
    # print len(arr),arr[0],arr[0][0]
    for i in range(0, len(arr)):
        worksheet.write(i, 0, arr[i][0])
    workbook.close()
    # import arcpy
    # path = 'F:/Test/Paper180829/Data/DEM/'
    outPoints = path + 'UTMDEMRG3840' + '.shp'
    arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))
    # print arr
    # np.savetxt(path + '1km_UTMLTN_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')
    outExcels = path + '10kmDEM' + '.xlsx'
    workbook = xlsxwriter.Workbook(outExcels)
    worksheet = workbook.add_worksheet('10kmDEM')
    # print len(arr),arr[0],arr[0][0]
    for i in range(0, len(arr)):
        worksheet.write(i, 0, arr[i][0])
    workbook.close()

    outPoints = path + 'UTMDEMRG384000.shp'
    arr = arcpy.da.FeatureClassToNumPyArray(outPoints, ('RASTERVALU'))
    # np.savetxt(path + '1km_UTMLTN_Txts/' +raster[0:9]+'.txt',arr,fmt = '%.8f')
    outExcels = path + '1kmDEM' + '.xlsx'
    workbook = xlsxwriter.Workbook(outExcels)
    worksheet = workbook.add_worksheet('1kmDEM')
    # print len(arr),arr[0],arr[0][0]
    for i in range(0, len(arr)):
        worksheet.write(i, 0, arr[i][0])
    workbook.close()


#   对IMERG/DEM/LTD/LTN进行归一化，合并到全部
# 合并
def f10():
    path = 'F:/Test/Paper180829/Test/'
    shpAll = 'F:/Test/Paper180829/Test/UTMPOINTS3840.shp'
    A = "shpAll"
    B = "shpIMERG"
    C = "shpDEM"
    D = "shpLTD"
    E = "shpLTN"
    arcpy.MakeFeatureLayer_management(shpAll, A)

    pathIMERG = 'F:/Test/Paper180829/Test/10kmRG3840/'
    pathDEM = 'F:/Test/Paper180829/Data/DEM/'
    pathLTD = 'F:/Test/Paper180829/Data/LTD/10km_UTMLTD_RG3840/'
    pathLTN = 'F:/Test/Paper180829/Data/LTN/10km_UTMLTN_RG3840/'
    for y in range(15, 18):
        for m in range(1, 13):
            shpIMERG = pathIMERG + 'I' + str((2000 + y) * 100 + m) + '.shp'
            shpDEM = pathDEM + 'UTMDEMRG3840' + '.shp'
            shpLTD = pathLTD + 'LTD' + str((2000 + y) * 100 + m) + '.shp'
            shpLTN = pathLTN + 'LTN' + str((2000 + y) * 100 + m) + '.shp'

            B = "shpIMERG" + str((2000 + y) * 100 + m)
            C = "shpDEM" + str((2000 + y) * 100 + m)
            D = "shpLTD" + str((2000 + y) * 100 + m)
            E = "shpLTN" + str((2000 + y) * 100 + m)

            arcpy.MakeFeatureLayer_management(shpIMERG, B)
            arcpy.MakeFeatureLayer_management(shpDEM, C)
            arcpy.MakeFeatureLayer_management(shpLTD, D)
            arcpy.MakeFeatureLayer_management(shpLTN, E)

            arcpy.AddJoin_management(A, "FID", B, "FID")
            arcpy.CalculateField_management(A, 'UTMPOINTS3840.IMERG',
                                            "!" + "I" + str((2000 + y) * 100 + m) + ".RASTERVALU!", "PYTHON_9.3")

            arcpy.AddJoin_management(A, "FID", C, "FID")
            arcpy.CalculateField_management(A, 'UTMPOINTS3840.DEM', "!" + "UTMDEMRG3840" + ".RASTERVALU!", "PYTHON_9.3")

            arcpy.AddJoin_management(A, "FID", D, "FID")
            arcpy.CalculateField_management(A, 'UTMPOINTS3840.LTD',
                                            "!" + "LTD" + str((2000 + y) * 100 + m) + ".RASTERVALU!",
                                            "PYTHON_9.3")
            arcpy.AddJoin_management(A, "FID", E, "FID")
            arcpy.CalculateField_management(A, 'UTMPOINTS3840.LTN',
                                            "!" + "LTN" + str((2000 + y) * 100 + m) + ".RASTERVALU!",
                                            "PYTHON_9.3")

            arcpy.RemoveJoin_management(A)

            arcpy.CalculateField_management(A, 'X', "!POINT_X!", "PYTHON_9.3")
            arcpy.CalculateField_management(A, 'Y', "!POINT_Y!", "PYTHON_9.3")

            outShp = path + '10kmData/' + 'd' + str((2000 + y) * 100 + m) + '.shp'
            # Execute Copy
            arcpy.CopyFeatures_management(A, outShp)

# 归一化
def f11():
    path = 'F:/Test/Paper180829/Test/'
    # arcpy.env.workspace = path+'10kmData/'
    # features=arcpy.ListFeatureClasses()
    for y in range(15, 18):
        for m in range(1, 13):

            shpData = path + '10kmData/' + 'd' + str((2000 + y) * 100 + m) + '.shp'
            F = "shpData" + str((2000 + y) * 100 + m)
            arcpy.MakeFeatureLayer_management(shpData, F)

            arrOrigin = arcpy.da.FeatureClassToNumPyArray(shpData, (
            'POINT_X', 'POINT_Y', 'IMERG', 'DEM', 'LTD', 'LTN', 'X', 'Y'))
            rows = len(arrOrigin)
            # print rows
            arrFinal = np.zeros((rows, 8), np.float)
            for i in range(0, rows):
                for j in range(0, 8):
                    arrFinal[i, j] = arrOrigin[i][j]
            # print arrFinal

            for j in range(2, 8):
                for i in range(0, rows):
                    arrOrigin[i][j] = (arrFinal[i, j] - min(arrFinal[:, j])) / (
                                max(arrFinal[:, j]) - min(arrFinal[:, j]))
            # print arrFinal
            # print arrOrigin
            outFC = path + '10kmNorm/' + 'n' + str((2000 + y) * 100 + m) + '.shp'
            arrOut = np.array(arrOrigin, np.dtype(
                [('POINT_X', np.float64), ('POINT_Y', np.float64), ('IMERG', np.float64), ('DEM', np.float64),
                 ('LTD', np.float64), ('LTN', np.float64), ('X', np.float64), ('Y', np.float64)]))
            SR = arcpy.Describe(shpData).spatialReference
            arcpy.da.NumPyArrayToFeatureClass(arrOut, outFC, ('POINT_X', 'POINT_Y'), SR)

            # arcpy.CalculateField_management(F, 'X', "!POINT_X!", "PYTHON_9.3")
            # arcpy.CalculateField_management(F, 'Y', "!POINT_Y!", "PYTHON_9.3")
        # arrAll = arcpy.da.FeatureClassToNumPyArray(fea,("ORIG_FID","POINT_X","POINT_Y"))
        # print arrAll[1][0]

# 探索性回归分析
def f12():
    path = 'F:/Test/Paper180829/Test/'
    # arcpy.env.workspace = path+'10kmData/'
    # features=arcpy.ListFeatureClasses()
    for y in range(15,18):
        for m in range(1,13):

            shpData = path +'10kmNorm/'+ 'n' + str((2000 + y) * 100 + m) + '.shp'
            G = "shpData" + str((2000 + y) * 100 + m)
            arcpy.MakeFeatureLayer_management(shpData, G)

            # swmFile=path +'10kmSWM/'+ 'swm' + str((2000 + y) * 100 + m) + '.swm'
            # swm = arcpy.GenerateSpatialWeightsMatrix_stats(shpData, "FID", swmFile,\
            #                                                "K_NEAREST_NEIGHBORS" )
            result=path +'10kmResult/'+ 'r' + str((2000 + y) * 100 + m) + '.txt'
            er = arcpy.ExploratoryRegression_stats(G, "IMERG",\
                                                   "DEM;LTD;LTN;X;Y",
                                                   '', result)

# "东西部分区实验，先要将数据分开，另制作合并的，归一化的shp,以便批量分析。"
def f13():
    rasDEM = Raster('F:/Test/Paper180829/Data/DEM/' + '10km_UTMHB8048_DEM.tif')
    rows, cols = rasDEM.height, rasDEM.width

    myDtype = {'names': ('ORIG_FID', 'POINT_X', 'POINT_Y', 'IMERG', 'DEM', 'LTD', 'LTN', 'X', 'Y'), 'formats': (
        np.long, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64)}
    arr = np.zeros(rows * cols / 2, myDtype)

    # ---------West Area------------------
    shpID_XY = 'F:/Test/Paper180829/Data/POINT/' + 'UTMPOINTS' + 'West' + '.shp'
    temp = arcpy.da.FeatureClassToNumPyArray(shpID_XY, ('ORIG_FID', 'POINT_X', 'POINT_Y'))

    ##  前三列放置ID，POINT_X,POINT_Y
    arr['ORIG_FID'] = temp['ORIG_FID']
    arr['POINT_X'] = temp['POINT_X']
    arr['POINT_Y'] = temp['POINT_Y']

    #   预留arr['IMERG']
    temp = arcpy.RasterToNumPyArray(rasDEM)
    arr['DEM'] = (temp[:, 0:cols / 2].copy()).reshape(rows * cols / 2)  # DEM
    #   预留arr['LTD']
    #   预留arr['LTN']
    arr['X'] = arr['POINT_X']  # X
    arr['Y'] = arr['POINT_Y']  # Y

    pathIMERG = 'F:/Test/Paper180829/Data/IMERG/10km8048/'
    pathLTD = 'F:/Test/Paper180829/Data/LTD/10km_UTMLTD_HB8048/'
    pathLTN = 'F:/Test/Paper180829/Data/LTN/10km_UTMLTN_HB8048/'
    path = 'F:/Test/Paper180829/Process/'

    for y in range(15, 18):
        for m in range(1, 13):
            name = str((2000 + y) * 100 + m)
            rasIMERG = Raster(pathIMERG + 'I' + name + '.tif')
            rasLTD = Raster(pathLTD + 'LTD' + name + '.tif')
            rasLTN = Raster(pathLTN + 'LTN' + name + '.tif')

            temp = arcpy.RasterToNumPyArray(rasIMERG)
            arr['IMERG'] = (temp[:, 0:cols / 2].copy()).reshape(rows * cols / 2)  # IMERG
            temp = arcpy.RasterToNumPyArray(rasLTD)
            arr['LTD'] = (temp[:, 0:cols / 2].copy()).reshape(rows * cols / 2)  # LTD
            temp = arcpy.RasterToNumPyArray(rasLTN)
            arr['LTN'] = (temp[:, 0:cols / 2].copy()).reshape(rows * cols / 2)  # LTN

            # print arr[1000]
            outFC = path + '10kmDataWest/' + 'd' + name + '.shp'
            SR = arcpy.Describe(shpID_XY).spatialReference
            arcpy.da.NumPyArrayToFeatureClass(arr, outFC, ('POINT_X', 'POINT_Y'), SR)
            print 'West Data ', name

            arr['IMERG'] = (arr['IMERG'] - arr['IMERG'].min()) / (arr['IMERG'].max() - arr['IMERG'].min())
            arr['DEM'] = (arr['DEM'] - arr['DEM'].min()) / (arr['DEM'].max() - arr['DEM'].min())
            arr['LTD'] = (arr['LTD'] - arr['LTD'].min()) / (arr['LTD'].max() - arr['LTD'].min())
            arr['LTN'] = (arr['LTN'] - arr['LTN'].min()) / (arr['LTN'].max() - arr['LTN'].min())
            arr['X'] = (arr['X'] - arr['X'].min()) / (arr['X'].max() - arr['X'].min())
            arr['Y'] = (arr['Y'] - arr['Y'].min()) / (arr['Y'].max() - arr['Y'].min())

            outFC = path + '10kmNormWest/' + 'n' + name + '.shp'
            SR = arcpy.Describe(shpID_XY).spatialReference
            arcpy.da.NumPyArrayToFeatureClass(arr, outFC, ('POINT_X', 'POINT_Y'), SR)
            print 'West Norm', name

    # ---------East Area------------------
    # shpID_XY = 'F:/Test/Paper180829/Data/POINT/' + 'UTMPOINTS' + 'East' + '.shp'
    # temp = arcpy.da.FeatureClassToNumPyArray(shpID_XY, ('ORIG_FID', 'POINT_X', 'POINT_Y'))
    #
    # ##  前三列放置ID，POINT_X,POINT_Y
    # arr['ORIG_FID'] = temp['ORIG_FID']
    # arr['POINT_X'] = temp['POINT_X']
    # arr['POINT_Y'] = temp['POINT_Y']
    #
    # #   预留arr['IMERG']
    # temp = arcpy.RasterToNumPyArray(rasDEM)
    # arr['DEM'] = (temp[:, cols/2:cols].copy()).reshape(rows * cols / 2)  # DEM
    # #   预留arr['LTD']
    # #   预留arr['LTN']
    # arr['X'] = arr['POINT_X']  # X
    # arr['Y'] = arr['POINT_Y']  # Y
    #
    # pathIMERG = 'F:/Test/Paper180829/Data/IMERG/10km8048/'
    # pathLTD = 'F:/Test/Paper180829/Data/LTD/10km_UTMLTD_HB8048/'
    # pathLTN = 'F:/Test/Paper180829/Data/LTN/10km_UTMLTN_HB8048/'
    # path = 'F:/Test/Paper180829/Process/'
    #
    # for y in range(15, 18):
    #     for m in range(1, 13):
    #         name = str((2000 + y) * 100 + m)
    #         rasIMERG = Raster(pathIMERG + 'I' + name + '.tif')
    #         rasLTD = Raster(pathLTD + 'LTD' + name + '.tif')
    #         rasLTN = Raster(pathLTN + 'LTN' + name + '.tif')
    #
    #         temp = arcpy.RasterToNumPyArray(rasIMERG)
    #         arr['IMERG'] = (temp[:, cols/2:cols].copy()).reshape(rows * cols / 2)  # IMERG
    #         temp = arcpy.RasterToNumPyArray(rasLTD)
    #         arr['LTD'] = (temp[:, cols/2:cols].copy()).reshape(rows * cols / 2)  # LTD
    #         temp = arcpy.RasterToNumPyArray(rasLTN)
    #         arr['LTN'] = (temp[:, cols/2:cols].copy()).reshape(rows * cols / 2)  # LTN
    #
    #         # print arr[1000]
    #         outFC = path + '10kmDataEast/' + 'd' + name + '.shp'
    #         SR = arcpy.Describe(shpID_XY).spatialReference
    #         arcpy.da.NumPyArrayToFeatureClass(arr, outFC, ('POINT_X', 'POINT_Y'), SR)
    #         print 'East Data ', name
    #
    #         arr['IMERG'] = (arr['IMERG'] - arr['IMERG'].min()) / (arr['IMERG'].max() - arr['IMERG'].min())
    #         arr['DEM'] = (arr['DEM'] - arr['DEM'].min()) / (arr['DEM'].max() - arr['DEM'].min())
    #         arr['LTD'] = (arr['LTD'] - arr['LTD'].min()) / (arr['LTD'].max() - arr['LTD'].min())
    #         arr['LTN'] = (arr['LTN'] - arr['LTN'].min()) / (arr['LTN'].max() - arr['LTN'].min())
    #         arr['X'] = (arr['X'] - arr['X'].min()) / (arr['X'].max() - arr['X'].min())
    #         arr['Y'] = (arr['Y'] - arr['Y'].min()) / (arr['Y'].max() - arr['Y'].min())
    #
    #         outFC = path + '10kmNormEast/' + 'n' + name + '.shp'
    #         SR = arcpy.Describe(shpID_XY).spatialReference
    #         arcpy.da.NumPyArrayToFeatureClass(arr, outFC, ('POINT_X', 'POINT_Y'), SR)
    #         print 'East Norm', name

# "把武汉点位row, col = 302, 556处GWR和MF降尺度结果导出便于Origin作图。"
def f14():

    path = 'F:/Test/Paper180829/MXD/Data/DownscaleResults/'
    workbook = xlsxwriter.Workbook(path + 'XYProfileAtWuhanStation' + '.xlsx')
    worksheet = workbook.add_worksheet('XYProfile')

    file = path + 'RectGWR201604.tif'
    print file
    ras = Raster(file)
    row, col = 302, 556
    arr = arcpy.RasterToNumPyArray(ras)
    rp, cp = arr.shape
    for r in range(row - 1, row):
        for c in range(0, cp):
            worksheet.write(c, 0, arr[r, c])
    for r in range(0, rp):
        for c in range(col - 1, col):
            worksheet.write(r, 1, arr[r, c])

    file = path + 'RectMF201604.tif'
    print file
    ras = Raster(file)
    row, col = 302, 556
    arr = arcpy.RasterToNumPyArray(ras)
    rp, cp = arr.shape
    for r in range(row - 1, row):
        for c in range(0, cp):
            worksheet.write(c, 2, arr[r, c])
    for r in range(0, rp):
        for c in range(col - 1, col):
            worksheet.write(r, 3, arr[r, c])
    workbook.close()