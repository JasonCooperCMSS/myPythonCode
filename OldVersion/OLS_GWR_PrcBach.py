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


#   计算误差损失
def scoreAdaptive(bdwt, y, x, east, north, i, kernelName='Bi_square', kernelType='Adaptive'):
    dx = east - east[i]
    dy = north - north[i]
    d = np.sqrt(dx * dx + dy * dy)

    print bdwt

    if kernelName == 'Gaussian':
        # print 'hahahaha'
        nRnd = np.where((d <= np.max(d)))
        nCounts = len(nRnd[0])
        # print nRnd[0]
        # print d[nRnd]
        wt = np.exp(- ((d[nRnd] / bdwt) ** 2))
        # wt[i]=0
        # wtt = np.exp(- (d[nRnd] / bdwt) ** 2)
        # nPos=np.where(wtt>0.01)
        # wt=wtt[nPos]
        w = np.mat(np.diag(wt))
        # print w
    elif kernelName == 'Exponential':
        print 'Exponential'
    elif kernelName == 'Bi_square':
        nRnd = np.where((d <= bdwt) & (d > 0))  # &(d>0)
        nCounts = len(nRnd[0])
        # print nRnd[0]
        # print d[nRnd]
        wt = (1 - (d[nRnd] / bdwt) ** 2) ** 2

        w = np.mat(np.diag(wt))
        # print w
    else:  # 'Tri_cube'
        print 'Tri_cube'

    xx = np.mat(x[nRnd, :])
    yy = np.mat(y[nRnd]).T
    # print yy.shape[0]
    xpxi = (xx.T * w * xx).I
    # print xpxi
    bi = xpxi * xx.T * w * yy
    sPoint = xx * xpxi * xx.T * w  # Y=Sy    帽子矩阵s
    trs = np.trace(sPoint)

    #   Resid
    yyp = np.mat(x[i, :]) * bi
    resid = y[i] - yyp
    return float(abs(resid))

    # # R2
    # yyp = xx * bi
    # Rsquare= np.sum((yy - np.mean(yy)).T * (yyp - np.mean(yyp))) ** 2 / np.sum((yy - np.mean(yy)).T * (yy - np.mean(yy))) / np.sum(
    #         (yyp - np.mean(yyp)).T * (yyp - np.mean(yyp)))
    # return float(-Rsquare)

    # # CV
    # iresid = y[i]-np.mat(x[i, :]) * bi
    # yyp = xx * bi
    # sse = (yy-yyp).T*(yy-yyp)
    # CV=(sse-iresid**2)/nCounts
    # return float(CV)

    # # AIC
    # yyp = xx * bi
    # sse = (yy - yyp).T * (yy - yyp)
    # AIC=2*nCounts*np.log(sse/(nCounts-trs))+nCounts*np.log(2*np.pi)+nCounts*(nCounts+trs)/(nCounts-2-trs)
    # print float(AIC)
    # return float(AIC)


def scoreFixed(bdwt, y, x, north, east, kernelName='Bi_square', kernelType='Fixed'):
    n = x.shape[0]
    k = x.shape[1]
    yp = np.mat(np.zeros((n, 1), np.float))
    resid = np.mat(np.zeros((n, 1), np.float))
    for i in range(0, n):
        dx = east - east[i]
        dy = north - north[i]
        d = np.sqrt(dx * dx + dy * dy)
        # print d[i]

        print i
        print bdwt

        if kernelName == 'Gaussian':
            # print 'hahahaha'
            nRnd = np.where((d <= np.max(d)))
            nCounts = len(nRnd[0])
            print nRnd[0]
            # print np.sort(d[nRnd])
            # print d[i]
            # wt = np.exp(- ((d[nRnd] / bdwt) ** 2))
            # wt[i]=0
            wtt = np.exp(- (d[nRnd] / bdwt) ** 2)
            nRnd = np.where(wtt > 0.001)
            wt = wtt[nRnd]
            w = np.mat(np.diag(wt))
            # print w
        elif kernelName == 'Exponential':
            print 'Exponential'
        elif kernelName == 'Bi_square':
            print 'hahaha'
            nRnd = np.where((d <= bdwt))  # &(d>0)
            nCounts = len(nRnd[0])
            # print nRnd[0]
            # print np.sort(d[nRnd])
            wt = (1 - (d[nRnd] / bdwt) ** 2) ** 2
            w = np.mat(np.diag(wt))
            # print w
        else:  # 'Tri_cube'
            print 'Tri_cube'

        xx = np.mat(x[nRnd, :])
        yy = np.mat(y[nRnd]).T
        # print yy.shape[0]
        xpxi = (xx.T * w * xx).I
        # print xpxi
        bi = xpxi * xx.T * w * yy
        sPoint = xx * xpxi * xx.T * w  # Y=Sy    帽子矩阵s
        trs = np.trace(sPoint)

        #   Resid
        yp[i] = np.mat(x[i, :]) * bi
        resid[i] = yp[i] - y[i]

    # # R2
    # ym=np.mat(y).T
    # Rsquare= np.sum((ym - np.mean(ym)).T * (yp - np.mean(yp))) ** 2 / np.sum((ym - np.mean(ym)).T * (ym - np.mean(ym))) / np.sum(
    #         (yp - np.mean(yp)).T * (yp - np.mean(yp)))
    # return float(-Rsquare)

    # # CV
    # sse = resid.T*resid
    # CV=sse/n
    # return float(CV)

    # AIC
    sse = resid.T * resid
    AIC = 2 * n * np.log(sse / (n - k)) + n * np.log(2 * np.pi) + nCounts * (n + k) / (n - 2 - k)
    return float(AIC)


#   完成一次GWR模型构建
def GWR(y, x, east, north, kernelName='Bi_square', kernelType='Adaptive'):  # Adaptive

    #   参数个数验证
    print kernelName, kernelType
    #   数据行数验证

    n = x.shape[0]
    k = x.shape[1]
    bdwt = np.zeros((n, 1), np.float)
    if kernelType == 'Adaptive':
        for i in range(0, n):
            dx = east - east[i]
            dy = north - north[i]
            d = np.sqrt(dx * dx + dy * dy)
            ds = np.sort(d)

            if kernelName == 'Gaussian':
                # print 'Gaussian_Adaptive'
                bmin = 0
                bmax = ds[n - 1]
                print bmin, bmax
                b, resid, ieer, numfunc = fminbound(scoreAdaptive, bmin, bmax,
                                                    (y, x, east, north, i, kernelName, kernelType), 1e-5, 500, True, 1)
                bdwt[i] = b
                # print bdwt[i],resid,ieer,numfunc
            elif kernelName == 'Exponential':
                print 'Exponential'
            elif kernelName == 'Bi_square':
                print 'Bi_square'
                bmin = ds[k + 3]
                bmax = ds[n - 1]
                # print bmin,bmax
                b, resid, ieer, numfunc = fminbound(scoreAdaptive, bmin, bmax,
                                                    (y, x, east, north, i, kernelName, kernelType), 1e-5, 500, True, 1)
                bdwt[i] = b
                # print bdwt[i],resid,ieer,numfunc
            else:  # 'Tri_cube'
                print 'Tri_cube'
    else:  # Fixed
        # darr=np.zeros(((n*(n-1))/2,1),np.float)
        darr = []
        dmin = np.zeros((n, 1), np.float)
        dmax = np.zeros((n, 1), np.float)
        for i in range(0, n):
            dx = east - east[i]
            dy = north - north[i]
            d = np.sqrt(dx * dx + dy * dy)
            ds = np.sort(d)

            dmin[i] = ds[k + 3]
            dmax[i] = ds[n - 1]
            print i, dmin[i], dmax[i]

        if kernelName == 'Gaussian':
            bmin = np.max(np.sort(dmin))
            bmax = np.max(np.sort(dmax))
            print bmin, bmax
            b, resid, ieer, numfunc = fminbound(scoreFixed, bmin, bmax,
                                                (y, x, east, north, kernelName, kernelType), 1e-5, 500, True, 1)
            for i in range(0, n):
                bdwt[i] = b
            # bdwt = b
            print bdwt, resid, ieer, numfunc
        elif kernelName == 'Exponential':
            print 'Exponential'
        elif kernelName == 'Bi_square':
            print 'Bi_square_Fixed'
            bmin = np.max(np.sort(dmin))
            bmax = np.max(np.sort(dmax))
            print bmin, bmax
            b, resid, ieer, numfunc = fminbound(scoreFixed, bmin, bmax,
                                                (y, x, east, north, kernelName, kernelType), 1e-5, 500, True, 1)
            for i in range(0, n):
                bdwt[i] = b
            # bdwt = b
            print bdwt, resid, ieer, numfunc
        else:  # 'Tri_cube'
            print 'Tri_cube'

    resultPoints = np.zeros((n, k + 2 + 1 + k + 1 + 2 + 2), np.float)
    resultPoints[:, 0] = y  # 原始y值
    resultPoints[:, 1:k] = x[:, 1:k]  # 原始x值
    resultPoints[:, k] = north[:]  # 原始南北坐标
    resultPoints[:, k + 1] = east[:]  # 原始东西坐标

    for i in range(0, n):

        resultPoints[i, k + 2] = bdwt[i]  # 阈值

        dx = east - east[i]
        dy = north - north[i]
        d = np.sqrt((dx * dx + dy * dy))

        if kernelName == 'Gaussian':
            nRnd = np.where((d <= np.max(d)))
            nCounts = len(nRnd[0])
            print nRnd[0]
            print np.sort(d[nRnd])
            # wt = np.exp(- (d[nRnd] / bdwt[i]) ** 2)
            # wt[i]=0
            wtt = np.exp(- (d[nRnd] / bdwt[i]) ** 2)
            print wtt
            nRnd = np.where((wtt > 0.001))
            print nRnd[0]
            wt = wtt[nRnd]
            w = np.mat(np.diag(wt))
            # print w
        elif kernelName == 'Exponential':
            print 'Exponential'
        elif kernelName == 'Bi_square':
            print 'Bi_square_Fixed'
            nRnd = np.where((d <= bdwt[i]) & (d > 0))  # &(d>0)
            nCounts = len(nRnd[0])
            print nCounts
            print np.sort(d[nRnd])
            wt = (1 - (d[nRnd] / bdwt[i]) ** 2) ** 2
            # wt[0]=0
            w = np.mat(np.diag(wt))
            # print w
        else:  # 'Tri_cube'
            print 'Tri_cube'

        xx = np.mat(x[nRnd, :])
        yy = np.mat(y[nRnd]).T

        xpxi = (xx.T * w * xx).I
        bi = xpxi * xx.T * w * yy
        sPoint = xx * xpxi * xx.T * w  # Y=Sy    帽子矩阵s
        trs = np.trace(sPoint)

        resultPoints[i, k + 3:2 * k + 3] = bi.T  # 系数

        #   预测值    残差
        yhat = np.mat(x[i, :]) * bi
        resid = np.mat(x[i, :]) * bi - y[i]
        resultPoints[i, 2 * k + 3] = yhat  # 预测值
        resultPoints[i, 2 * k + 4] = resid  # 残差

        #   当前点拟合误差
        ssePoint = (yy - xx * bi).T * (yy - xx * bi)  # SSE
        #   R方和校正R方
        yyp = xx * bi
        Rsquare = np.sum((yy - np.mean(yy)).T * (yyp - np.mean(yyp))) ** 2 / (
                    (yy - np.mean(yy)).T * (yy - np.mean(yy))) / ((yyp - np.mean(yyp)).T * (yyp - np.mean(yyp)))
        resultPoints[i, 2 * k + 5] = Rsquare  # R方
        # print resultPoints[i, 2 * k + 5]
        # print np.sum((yy-np.mean(yy)).T*(yyp-np.mean(yyp)))**2/((yy-np.mean(yy)).T*(yy-np.mean(yy)))/((yyp-np.mean(yyp)).T*(yyp-np.mean(yyp)))

        resultPoints[i, 2 * k + 6] = (ssePoint) / nCounts  # CV
        resultPoints[i, 2 * k + 7] = 2 * nCounts * np.log(ssePoint / (nCounts - trs)) + nCounts * np.log(
            2 * np.pi) + nCounts * (nCounts + trs) / (nCounts - 2 - trs)  # AIC

    resultModel = np.zeros((6, 1), np.float)
    sseModel = np.sum(resultPoints[:, 2 * k + 4] ** 2)  # SSE of MODEL
    # resultModel[0]=1-sseModel/sstModel    #   R方

    ypm = np.mat(resultPoints[:, 2 * k + 3]).T
    ym = np.mat(y).T
    resultModel[0] = np.sum((ym - np.mean(ym)).T * (ypm - np.mean(ypm))) ** 2 / (
                (ym - np.mean(ym)).T * (ym - np.mean(ym))) / ((ypm - np.mean(ypm)).T * (ypm - np.mean(ypm)))
    # print resultModel[0]
    resultModel[1] = np.sum(resultPoints[:, 2 * k + 3]) / np.sum(y) - 1  # Bias
    resultModel[2] = np.sqrt(np.sum((resultPoints[:, 2 * k + 4] ** 2)) / n)  # RMSE
    resultModel[3] = np.sum(np.abs(resultPoints[:, 2 * k + 4])) / n  # MAE
    resultModel[4] = (sseModel) / n  # GCV    广义交叉验证法
    resultModel[5] = 2 * n * np.log(sseModel / (n - k)) + n * np.log(2 * np.pi) + n * (n + k) / (n - 2 - k)  # AIC

    # print resultPoints,resultModel
    return resultPoints, resultModel


#   主函数完成，数据读入和批处理循环

def MLR(y, x, east, north):
    #   数据行数验证

    n = x.shape[0]
    k = x.shape[1]
    xt = np.zeros((n, k + 2), np.float)
    xt[:, 0:k] = x
    xt[:, k] = east
    xt[:, k + 1] = north
    xx = np.mat(xt)
    yy = np.mat(y).T

    xpxi = (xx.T * xx).I
    bi = xpxi * xx.T * yy

    yhat = xx * bi
    resid = yy - yhat

    resultPoints = np.zeros((n, k + 2+k+2+2), np.float)
    resultPoints[:, 0:1] = yy  # 原始y值
    resultPoints[:, 1:k + 2] = xx[:, 1:k + 2]  # 原始自变量
    for r in range(0,n):
        resultPoints[r, k+2:k+2+k+2] = bi.T  # 系数
    resultPoints[:, k + 2+k+2:k + 2+k+3] = yhat  # 拟合值
    resultPoints[:, k + 2+k+3:k + 2+k+4] = resid  # 残差

    resultModel = np.zeros((6, 1), np.float)
    sseModel = resid.T * resid
    resultModel[0] = np.sum((yy - np.mean(yy)).T * (yhat - np.mean(yhat))) ** 2 / (
            (yy - np.mean(yy)).T * (yy - np.mean(yy))) / ((yhat - np.mean(yhat)).T * (yhat - np.mean(yhat)))
    # print resultModel[0]
    resultModel[1] = np.sum(yhat) / np.sum(y) - 1  # Bias
    resultModel[2] = np.sqrt(sseModel / n)  # RMSE
    resultModel[3] = np.sum(np.abs(resid)) / n  # MAE
    resultModel[4] = (sseModel) / n  # GCV    广义交叉验证法
    resultModel[5] = 2 * n * np.log(sseModel / (n - k)) + n * np.log(2 * np.pi) + n * (n + k) / (n - 2 - k)  # AIC

    print resultPoints, resultModel
    return resultPoints, resultModel


def main():
    path = 'F:/Test/GWR/Point_OLS_GWR/'
    file = path + 'Sample.xlsx'
    excelFile = xlrd.open_workbook(file)
    t = excelFile.sheet_by_index(0)
    rows = t.nrows
    cols = t.ncols

    d = np.zeros((rows, cols), np.float)
    for c in range(0, cols):
        d[:, c] = t.col_values(c, 0, rows)

    y = d[:, 0].copy()
    x = d[:, 0:cols-2].copy()
    x[:, 0] = 1
    east = d[:, cols-2].copy()
    north = d[:, cols - 1].copy()

    n = x.shape[0]
    k = x.shape[1]
    kernelName = 'Gaussian'  # Gaussian Exponential  Bi_square   Tri_cube
    kernelType = 'Fixed'  # Fixed  Adaptive
    # resultPoint, resultModel = GWR(y, x, east, north, kernelName, kernelType)
    resultPoint, resultModel =MLR(y, x, east, north)

    workbook = xlsxwriter.Workbook(path + 'r/' + 'out1.xlsx')  # out2样本点本身权重为1
    worksheet1 = workbook.add_worksheet('points_Bi_square_Adaptive')
    worksheet2 = workbook.add_worksheet('model_Bi_square_Adaptive')

    # for r in range(0, n):
    #     for c in range(0, 2 * k + 8):
    #         worksheet1.write(r, c, resultPoint[r, c])
    for r in range(0,n):
        for c in range(0,k+4):
             worksheet1.write(r, c, resultPoint[r,c])

    for r in range(0, 6):
        worksheet2.write(r, 0, resultModel[r])
    workbook.close()
main()