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

arcpy.env.overwriteOutput = True


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
    return y, x


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
    return xPred


def fit(y, x, name, pathOut):
    n, k = x.shape
    xx, yy = mat(x), mat(y)

    xpxi = (xx.T * xx).I
    bi = xpxi * xx.T * yy
    yp = xx * bi
    resid = yy - yp

    sPoint = xx * xpxi * xx.T  # 帽子矩阵
    trs = k  # 帽子矩阵的迹

    sse = resid.T * resid
    sst = (yy - mean(yy)).T * (yy - mean(yy))  # 总离差平方和
    sigmaSquare = sse / (n - trs)  # 误差平方和的估计值

    sample = Raster('F:/Test/Paper180829/Data/IMERG/10km8048/' + 'I' + '201501' + '.tif')
    rows, cols = sample.height, sample.width
    lowerLeft = arcpy.Point(sample.extent.XMin, sample.extent.YMin)
    cellSize = sample.meanCellWidth

    # Set environmental variables for output
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = sample
    arcpy.env.cellSize = sample

    pathFit = pathOut
    arr = where((yp < 0), 0, yp)
    rasFitted = arcpy.NumPyArrayToRaster(arr.reshape((rows, cols)).getA(), lowerLeft, cellSize, cellSize)
    rasFitted.save(pathFit + 'Fitted_' + 'MLR_' + name + '.tif')

    rasResid = arcpy.NumPyArrayToRaster(resid.reshape((rows, cols)).getA(), lowerLeft, cellSize, cellSize)
    rasResid.save(pathFit + 'Resid_' + 'MLR_' + name + '.tif')

    rasStdResid = arcpy.NumPyArrayToRaster(
        (resid / (sqrt(sigmaSquare * (1 - diag(sPoint)))).T).reshape((rows, cols)).getA(), lowerLeft, cellSize,
        cellSize)
    rasStdResid.save(pathFit + 'StdResid_' + 'MLR_' + name + '.tif')

    resultModel = zeros((7 + k, 1), float)
    resultModel[0] = 1 - sse / sst  # R2
    resultModel[1] = 1 - (sse / (n - trs)) / (sst / (n - 1))  # 校正R2
    resultModel[2] = 2 * n * log(sse / (n - trs)) + n * log(2 * pi) + n * (n + trs) / (n - 2 - trs)  # AICc
    resultModel[3] = sum((yy - mean(yy)).T * (yp - mean(yp))) ** 2 / \
                     ((yy - mean(yy)).T * (yy - mean(yy))) / ((yp - mean(yp)).T * (yp - mean(yp)))  # 相关系数
    resultModel[4] = sum(yp) / sum(y) - 1  # Bias
    resultModel[5] = sqrt(sse / n)  # RMSE
    resultModel[6] = sum(abs(resid)) / n  # MAE
    resultModel[7:7 + k] = bi

    workbook = xlsxwriter.Workbook(pathFit + 'modelFitness_' + 'MLR_' + name + '.xlsx')  # modelFitness
    worksheet = workbook.add_worksheet('modelFitness_' + 'MLR_' + name)
    rp, cp = resultModel.shape
    for r in range(0, rp):
        for c in range(0, cp):
            worksheet.write(r, c, resultModel[r, c])
    workbook.close()
    return bi


def pred(beta, xPred, name, pathOut):
    xo = mat(xPred)
    yPred = xo * beta
    arr = where((yPred < 0), 0, yPred)

    sample = Raster('F:/Test/Paper180829/Data/DEM/' + '1km_UTMHB8048_DEM.tif')
    rows, cols = sample.height, sample.width
    lowerLeft = arcpy.Point(sample.extent.XMin, sample.extent.YMin)
    cellSize = sample.meanCellWidth
    # Set environmental variables for output
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = sample
    arcpy.env.cellSize = sample

    pathPred = pathOut
    rasPred = arcpy.NumPyArrayToRaster(arr.reshape((rows, cols)), lowerLeft, cellSize, cellSize)  # .getA()
    rasPred.save(pathPred + 'Predicted_' + 'MLR_' + name + '.tif')


def main():
    st = time.clock()
    pathIn = 'F:/Test/Paper180829/Data/'
    pathOut = 'F:/Test/Paper180829/Process/' + '0/'
    for year in range(15, 16):
        for month in range(12, 13):
            name = str((2000 + year) * 100 + month)
            print name, '###################################'
            # fit data import
            y, x = FitDataImport(pathIn, name)
            # prediction data import
            xPred = PredDataImport(pathIn, name)
            # finish the fit and prediction
            t0 = time.clock()
            coef = fit(y, x, name, pathOut)
            t1 = time.clock()
            print "低分辨率下拟合耗时为:", t1 - t0, "秒。"
            result = pred(coef, xPred, name, pathOut)
            t2 = time.clock()
            print "高分辨率下拟合耗时为:", t2 - t1, "秒。"
            del y, x, xPred
            gc.collect()
            break
        break
    et = time.clock()
    print "总耗时为:", et - st, "秒。"


main()
