# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *

import sys, string, os
import xlrd
import xlwt
import xlsxwriter
from numpy import *
from scipy.optimize import fminbound
import time
import math
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import *
from dbfread import DBF
import gc


#   计算误差损失
def scoreFunc(bdwt, y, x, eastFit, northFit, kernelName='Bi_square', kernelType='Adaptive'):
    # print bdwt
    n, k = x.shape
    resid = mat(zeros((n, 1), float))
    trs = 0
    if kernelType == 'Adaptive':
        for i in range(0, n):
            dx = eastFit - eastFit[i]
            dy = northFit - northFit[i]
            d = sqrt(dx * dx + dy * dy)
            ds = sort(d.flat)

            if kernelName == 'Bi_square':
                nRnd = where((d <= ds[int(bdwt)]))  # bdwt
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                wt = (1 - (d[nRnd[0]] / ds[int(bdwt)]) ** 2) ** 2  # 这里用nRnd[0],因为d是n*1一维数组，得到的还是n*1数组。
                # print wt.shape
                w = mat(diag(wt.flat))  # 由于上面采用nRnd[0]，这里为了正确化为对角矩阵，需要先对wt展平
                # print w.shape
            else:  # kernelName == 'Gaussian'
                nRnd = where((d <= 4 * ds[int(bdwt)]))  # bdwt
                # 为什么是这里*4而不是下面权重计算是*4？是因为4倍距离内的点数不一定是当前带宽点数的4倍
                # 而点数4倍处的距离也不一定是当前点处距离的4倍，所以在自适应核中，在这一步*4
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                wt = exp(- ((d[nRnd[0]] / ds[int(bdwt)]) ** 2) / 2)
                w = mat(diag(wt.flat))
                # print w
            xx = mat(x[nRnd[0], :])  # 这里用nRnd[0],因为x是一个p*k数组，对它数组进行nRnd切片，会得到三维数组，（1，p，k)。
            yy = mat(y[nRnd[0]])
            # print i,bdwt,xx.shape,w.shape
            xpxi = (xx.T * w * xx).I
            # print xpxi
            bi = xpxi * xx.T * w * yy
            yp = xx[id[0], :] * bi
            resid[i] = yy[id[0]] - yp
            hat = xx[id[0], :] * xpxi * xx.T * w
            trs = trs + hat[0, id[0]]  # 巧妙地记录帽子矩阵
            # 帽子矩阵的秩，是对角线之和，在每一步中生成的是总的帽子矩阵的一行，
            # 所以在每一步中单独记下当前点在参与当前点回归的所有点中的序号，这个序号就是当前点在全部n个中恰好对应的
            # 以此起到不用记录整个帽子矩阵，节省内存

    else:  # Fixed
        for i in range(0, n):
            dx = eastFit - eastFit[i]
            dy = northFit - northFit[i]
            d = sqrt(dx * dx + dy * dy)

            if kernelName == 'Gaussian':
                # print 'Gaussian'
                # nRnd = where((d <= 4*bdwt))
                # id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                # wt = exp(- ((d[nRnd[0]] / bdwt) ** 2) / 2)
                nRnd = where((d <= bdwt))
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                wt = exp(- ((d[nRnd[0]] / bdwt * 4) ** 2) / 2)  # 而在固定核中，在这一步*4（分母带宽/4）
                w = mat(diag(wt.flat))
                # print w
            else:  # kernelName == 'Bi_square'
                # print 'Bi_square'
                nRnd = where((d <= bdwt))
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                wt = (1 - (d[nRnd[0]] / bdwt) ** 2) ** 2
                w = mat(diag(wt.flat))
                # print w
            xx = mat(x[nRnd[0], :])
            yy = mat(y[nRnd[0]])
            xpxi = (xx.T * w * xx).I
            # print xpxi
            bi = xpxi * xx.T * w * yy
            # print id[0]
            yp = xx[id[0], :] * bi
            resid[i] = yy[id[0]] - yp
            hat = xx[id[0], :] * xpxi * xx.T * w
            trs = trs + hat[0, id[0]]
            # 原理同上
    # AICc
    sse = resid.T * resid
    AICc = 2 * n * log(sse / (n - trs)) + n * log(2 * pi) + n * (n + trs) / (n - 2 - trs)
    # print AICc
    return float(AICc)


#   最优带宽搜索
def setBdwt(y, x, eastFit, northFit, kernelType, kernelName):
    # ---------------------寻找最佳阈值
    n, k = x.shape
    if kernelType == 'Adaptive':
        print '当前选择核函数类型是：Adaptive'
        bmin = k
        bmax = 100
    else:  # Fixed
        print '当前选择核函数类型是：Fixed'
        dmin = zeros((n, 1), float)
        dmax = zeros((n, 1), float)
        for i in range(0, n):
            dx = eastFit - eastFit[i]
            dy = northFit - northFit[i]
            d = sqrt(dx * dx + dy * dy)
            # print d.shape
            ds = sort(d.flat)
            dmin[i] = ds[k]
            dmax[i] = ds[100]
            # print i, dmin[i], dmax[i]
        bmin = max(sort(dmin))  # max(sort(dmin))
        bmax = max(sort(dmax))
    print '对应阈值搜索范围是：', bmin, bmax
    bdwt, result, ieer, numfunc = fminbound(scoreFunc, bmin, bmax, (y, x, eastFit, northFit, kernelName, kernelType), 3,
                                            500, True, 1)
    print '阈值搜索结果：', bdwt, result, ieer, numfunc
    return bdwt


#   完成拟合
def fitGWR(y, x, eastFit, northFit, kernelType, kernelName, bdwt, name, pathOut):
    n, k = x.shape
    bi = mat(zeros((n, k), float))
    yp = mat(zeros((n, 1), float))
    resid = mat(zeros((n, 1), float))
    # 训练阶段
    trsRow = mat(zeros((n, 1), float))
    if kernelType == 'Adaptive':
        for i in range(0, n):
            dx = eastFit - eastFit[i]
            dy = northFit - northFit[i]
            d = sqrt(dx * dx + dy * dy)
            ds = sort(d.flat)

            if kernelName == 'Bi_square':
                # print 'Bi_square'
                nRnd = where((d <= ds[int(bdwt)]))
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                wt = (1 - (d[nRnd[0]] / ds[int(bdwt)]) ** 2) ** 2
                w = mat(diag(wt.flat))
                # print w
            else:  # kernelName == 'Gaussian'
                # print 'Gaussian'
                nRnd = where((d <= 4 * ds[int(bdwt)]))
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                wt = exp(- ((d[nRnd[0]] / ds[int(bdwt)]) ** 2) / 2)
                w = mat(diag(wt.flat))
                # print w
            xx = mat(x[nRnd[0], :])
            yy = mat(y[nRnd[0]])
            xpxi = (xx.T * w * xx).I
            # print xpxi
            bi[i, :] = (xpxi * xx.T * w * yy).T
            yp[i] = xx[id[0], :] * bi[i, :].T
            resid[i] = yy[id[0]] - yp[i]
            hat = xx[id[0], :] * xpxi * xx.T * w
            trsRow[i] = hat[0, id[0]]
            # print xx[id,:]
            # print x[i,:]
    else:  # Fixed
        for i in range(0, n):
            dx = eastFit - eastFit[i]
            dy = northFit - northFit[i]
            d = sqrt(dx * dx + dy * dy)

            if kernelName == 'Gaussian':
                # nRnd = where((d <= 4*bdwt))
                # id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                # # print 'hahahaha'
                # wt = exp(- ((d[nRnd[0]] / bdwt) ** 2) / 2)
                nRnd = where((d <= bdwt))
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                # print 'hahahaha'
                wt = exp(- ((d[nRnd[0]] / bdwt * 4) ** 2) / 2)
                w = mat(diag(wt.flat))
                # print w
            else:  # kernelName == 'Bi_square'
                nRnd = where((d <= bdwt))
                id = where((d[nRnd[0]] == 0))  # 记录帽子矩阵斜对角元素在相对局部上的位置
                wt = (1 - (d[nRnd[0]] / bdwt) ** 2) ** 2
                w = mat(diag(wt.flat))
                # print w
            xx = mat(x[nRnd[0], :])
            yy = mat(y[nRnd[0]])
            xpxi = (xx.T * w * xx).I
            # print xpxi
            bi[i, :] = (xpxi * xx.T * w * yy).T
            yp[i] = xx[id[0], :] * bi[i, :].T
            resid[i] = yy[id[0]] - yp[i]
            hat = xx[id[0], :] * xpxi * xx.T * w
            trsRow[i] = hat[0, id[0]]  # 记录帽子矩阵元素
            # print xx[id, :]
            # print x[i, :]
            # print 'fitting ',i

    yo = mat(y)
    sse = resid.T * resid
    sst = (yo - mean(yo)).T * (yo - mean(yo))
    trs = sum(trsRow)
    # print trsRow
    # print trs

    # 输出拟合结果、残差、学生残差、系数等
    sample = Raster('F:/Test/Paper180829/Data/IMERG/10km8048/' + 'I' + '201501' + '.tif')
    rows, cols = sample.height, sample.width
    lowerLeft = arcpy.Point(sample.extent.XMin, sample.extent.YMin)
    cellSize = sample.meanCellWidth

    # Set environmental variables for output
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = sample
    arcpy.env.cellSize = sample

    arr = where((yp < 0), 0, yp)
    pathFit = 'F:/Test/Paper180829/Process/' + '0/'
    rasFitted = arcpy.NumPyArrayToRaster(arr.reshape((rows, cols)), lowerLeft, cellSize, cellSize)  # .getA()
    rasFitted.save(pathFit + 'Fitted_' + kernelName + '_' + kernelType + name + '.tif')

    rasResid = arcpy.NumPyArrayToRaster(resid.reshape((rows, cols)).getA(), lowerLeft, cellSize, cellSize)
    rasResid.save(pathFit + 'Resid_' + kernelName + '_' + kernelType + name + '.tif')

    rasStdResid = arcpy.NumPyArrayToRaster(
        (resid / (sqrt(sse / (n - trs) * (1 - trsRow).T)).T).reshape((rows, cols)).getA(), lowerLeft, cellSize,
        cellSize)
    rasStdResid.save(pathFit + 'StdResid_' + kernelName + '_' + kernelType + name + '.tif') \
        #
    # rasCoef = arcpy.NumPyArrayToRaster(
    #     (bi[:, 0:1]).reshape((rows, cols)).getA(), lowerLeft, cellSize,
    #     cellSize)
    # rasCoef.save(pathFit + 'Coef1_' + kernelName + '_' + kernelType + name + '.tif')
    # rasCoef = arcpy.NumPyArrayToRaster(
    #     (bi[:, 1:2]).reshape((rows, cols)).getA(), lowerLeft, cellSize,
    #     cellSize)
    # rasCoef.save(pathFit + 'Coef2_' + kernelName + '_' + kernelType + name + '.tif')
    # rasCoef = arcpy.NumPyArrayToRaster(
    #     (bi[:, 2:3]).reshape((rows, cols)).getA(), lowerLeft, cellSize,
    #     cellSize)
    # rasCoef.save(pathFit + 'Coef3_' + kernelName + '_' + kernelType + name + '.tif')
    # rasCoef = arcpy.NumPyArrayToRaster(
    #     (bi[:, 3:4]).reshape((rows, cols)).getA(), lowerLeft, cellSize,
    #     cellSize)
    # rasCoef.save(pathFit + 'Coef4_' + kernelName + '_' + kernelType + name + '.tif')
    # rasCoef = arcpy.NumPyArrayToRaster(
    #     (bi[:, 4:5]).reshape((rows, cols)).getA(), lowerLeft, cellSize,
    #     cellSize)
    # rasCoef.save(pathFit + 'Coef5_' + kernelName + '_' + kernelType + name + '.tif')

    # 输出模型拟合情况
    resultModel = zeros((8, 1), float)
    if (kernelType == 'Adaptive'):
        resultModel[0, 0] = int(bdwt)  # 阈值
    else:
        resultModel[0, 0] = bdwt / 4

    resultModel[1, 0] = 1 - sse / sst  # R2
    resultModel[2, 0] = 1 - (sse / (n - trs)) / (sst / (n - 1))  # 校正R2
    resultModel[3, 0] = 2 * n * log(sse / (n - trs)) + n * log(2 * pi) + n * (n + trs) / (n - 2 - trs)  # AICc
    resultModel[4, 0] = ((yo - mean(yo)).T * (yp - mean(yp))) ** 2 / ((yo - mean(yo)).T * (yo - mean(yo))) / (
            (yp - mean(yp)).T * (yp - mean(yp)))  # 相关系数
    resultModel[5, 0] = sum(yp) / sum(yo) - 1  # Bias
    resultModel[6, 0] = sqrt(sse / n)  # RMSE
    resultModel[7, 0] = sum(abs(resid)) / n  # MAE

    workbook = xlsxwriter.Workbook(
        pathFit + 'modelFitness_' + kernelName + '_' + kernelType + name + '.xlsx')
    worksheet = workbook.add_worksheet('modelFitness_' + kernelName + '_' + kernelType)
    rp, cp = resultModel.shape
    for r in range(0, rp):
        for c in range(0, cp):
            worksheet.write(r, c, resultModel[r, c])
    workbook.close()
    # print sse,trs
    # return resultFit,resultModel


#   完成预测
def predGWR(y, x, eastFit, northFit, kernelType, kernelName, bdwt, xPred, eastPred, northPred, name, pathOut):
    npre, kpre = xPred.shape
    n, k = x.shape

    if kpre != k:
        print 'xpreError'
        return [], []
    bi = mat(zeros((npre, k), float))
    yp = mat(zeros((npre, 1), float))
    xo = mat(xPred)
    if kernelType == 'Adaptive':
        for i in range(0, npre):
            dx = eastFit - eastPred[i]
            dy = northFit - northPred[i]
            d = sqrt(dx * dx + dy * dy)
            ds = sort(d.flat)

            if kernelName == 'Bi_square':
                # print 'Bi_square'
                nRnd = where((d <= ds[int(bdwt)]))
                wt = (1 - (d[nRnd[0]] / ds[int(bdwt)]) ** 2) ** 2
                w = mat(diag(wt.flat))
                # print w
            else:  # kernelName == 'Gaussian'
                # print 'Gaussian'
                nRnd = where((d <= 4 * ds[int(bdwt)]))
                wt = exp(- ((d[nRnd[0]] / ds[int(bdwt)]) ** 2) / 2)
                w = mat(diag(wt.flat))
                # print w
            xx = mat(x[nRnd[0], :])
            yy = mat(y[nRnd[0]])
            xpxi = (xx.T * w * xx).I
            # print xpxi
            bi[i, :] = (xpxi * xx.T * w * yy).T
            yp[i] = xo[i, :] * bi[i, :].T

            # print xx[id,:]
            # print x[i,:]
    else:  # Fixed
        for i in range(0, npre):
            dx = eastFit - eastPred[i]
            dy = northFit - northPred[i]
            d = sqrt(dx * dx + dy * dy)

            if kernelName == 'Gaussian':
                # nRnd = where((d <= 4 * bdwt))
                # # print 'hahahaha'
                # wt = exp(- ((d[nRnd[0]] / bdwt) ** 2) / 2)
                nRnd = where((d <= bdwt))
                # print 'hahahaha'
                wt = exp(- ((d[nRnd[0]] / bdwt * 4) ** 2) / 2)
                w = mat(diag(wt.flat))
                # print w
            else:  # kernelName == 'Bi_square'
                nRnd = where((d <= bdwt))
                wt = (1 - (d[nRnd[0]] / bdwt) ** 2) ** 2
                w = mat(diag(wt.flat))
                # print w
            xx = mat(x[nRnd[0], :])
            yy = mat(y[nRnd[0]])
            xpxi = (xx.T * w * xx).I
            # print xpxi
            bi[i, :] = (xpxi * xx.T * w * yy).T
            yp[i] = xo[i, :] * bi[i, :].T
            # print 'predicting:',i

            # print xx[id, :]
            # print x[i, :]
    # print trsRow
    # print trs

    sample = Raster('F:/Test/Paper180829/Data/DEM/' + '1km_UTMHB8048_DEM.tif')
    rows, cols = sample.height, sample.width
    lowerLeft = arcpy.Point(sample.extent.XMin, sample.extent.YMin)
    cellSize = sample.meanCellWidth
    # Set environmental variables for output
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = sample
    arcpy.env.cellSize = sample

    arr = where((yp < 0), 0, yp)
    pathPred = pathOut
    rasPred = arcpy.NumPyArrayToRaster(arr.reshape((rows, cols)), lowerLeft, cellSize, cellSize)  # .getA()
    rasPred.save(pathPred + 'Predicted_' + kernelName + '_' + kernelType + name + '.tif')


#   拟合数据导入
def FitDataImport(pathIn, name):
    fileIMERG = pathIn + 'IMERG/10km8048/' + 'I' + name + '.tif'
    IMERG = Raster(fileIMERG)
    rows, cols = IMERG.height, IMERG.width
    lowerLeft = arcpy.Point(IMERG.extent.XMin, IMERG.extent.YMin)
    cellSize = IMERG.meanCellWidth

    arr = arcpy.RasterToNumPyArray(IMERG, nodata_to_value=0)
    y = arr.reshape((rows * cols), 1).copy()

    x = ones((rows * cols, 5), float)
    fileDEM = pathIn + 'DEM/' + '10km_UTMHB8048_DEM.tif'
    DEM = Raster(fileDEM)
    arr = arcpy.RasterToNumPyArray(DEM, nodata_to_value=0)
    x[:, 1:2] = arr.reshape((rows * cols), 1).copy()

    fileLTD = pathIn + 'LTD/10km_UTMLTD_HB8048/' + 'LTD' + name + '.tif'
    LTD = Raster(fileLTD)
    arr = arcpy.RasterToNumPyArray(LTD, nodata_to_value=0)
    x[:, 2:3] = arr.reshape((rows * cols), 1).copy()

    for c in range(0, cols):
        arr[:, c] = lowerLeft.X + c * cellSize
    eastFit = arr.reshape((rows * cols), 1).copy()
    x[:, 3:4] = eastFit.copy()

    for r in range(0, rows):
        arr[r, :] = lowerLeft.Y + (rows - r) * cellSize
    northFit = arr.reshape((rows * cols), 1).copy()
    x[:, 4:5] = northFit.copy()

    print '因变量数组大小为：', y.shape, '；自变量数组大小为：', x.shape
    return y, x, eastFit, northFit


#   预测数据导入
def PredDataImport(pathIn, name):
    fileDEM = pathIn + 'DEM/' + '1km_UTMHB8048_DEM.tif'
    DEM = Raster(fileDEM)
    rows, cols = DEM.height, DEM.width
    lowerLeft = arcpy.Point(DEM.extent.XMin, DEM.extent.YMin)
    cellSize = DEM.meanCellWidth

    xPred = ones((rows * cols, 5), float)
    arr = arcpy.RasterToNumPyArray(DEM, nodata_to_value=0)
    xPred[:, 1:2] = arr.reshape((rows * cols), 1).copy()

    fileLTD = pathIn + 'LTD/1km_UTMLTD_HB8048/' + 'LTD' + name + '.tif'
    LTD = Raster(fileLTD)
    arr = arcpy.RasterToNumPyArray(LTD, nodata_to_value=0)
    xPred[:, 2:3] = arr.reshape((rows * cols), 1).copy()

    for c in range(0, cols):
        arr[:, c] = lowerLeft.X + c * cellSize
    eastPred = arr.reshape((rows * cols), 1).copy()
    xPred[:, 3:4] = eastPred.copy()

    for r in range(0, rows):
        arr[r, :] = lowerLeft.Y + (rows - r) * cellSize
    northPred = arr.reshape((rows * cols), 1).copy()
    xPred[:, 4:5] = northPred.copy()

    print '1km时自变量数组大小为：', xPred.shape
    return xPred, eastPred, northPred


#   主函数完成批处理循环
def main():
    st = time.clock()
    pathIn = 'F:/Test/Paper180829/Data/'
    pathOut = 'F:/Test/Paper180829/Process/' + '0/'
    kernelName = 'Gaussian'  # Gaussian Exponential  Bi_square   Tri_cube
    kernelType = 'Fixed'  # Fixed  Adaptive

    for year in range(15, 18):
        for month in range(1, 13):
            name = str((2000 + year) * 100 + month)
            print name, '###################################'
            # fit data import
            y, x, eastFit, northFit = FitDataImport(pathIn, name)
            # prediction data import
            xPred, eastPred, northPred = PredDataImport(pathIn, name)

            t0 = time.clock()
            bestBdwt = setBdwt(y, x, eastFit, northFit, kernelType, kernelName)
            t1 = time.clock()
            print "寻找最优带宽耗时为:", t1 - t0, "秒。"
            fitGWR(y, x, eastFit, northFit, kernelType, kernelName, bestBdwt, name, pathOut)
            t2 = time.clock()
            print "低分辨率下拟合耗时为:", t2 - t1, "秒。"
            predGWR(y, x, eastFit, northFit, kernelType, kernelName, bestBdwt, xPred, eastPred, northPred, name,
                    pathOut)
            t3 = time.clock()
            print "高分辨率下拟合耗时为:", t3 - t2, "秒。"
            del y, x, eastFit, northFit, xPred, eastPred, northPred
            gc.collect()

            # break
        # break
    et = time.clock()
    print "总耗时为:", et - st, "秒。"


main()
