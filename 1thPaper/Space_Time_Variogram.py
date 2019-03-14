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



def Space_Time_Variogram():
    path = 'F:/Test/Paper180614/Result/'
    d = xlrd.open_workbook(path+'rain_Origin.xls')

    #   控制计算范围

    t1 = d.sheet_by_index(0)    #0是RGS，1是IMERG
    rows,cols = 83,36
    x=t1.col_values(7, 1, rows+1)
    y=t1.col_values(8, 1, rows+1)
    for j in range(0, rows):
        x[j] = float(x[j])
        y[j] = float(y[j])
    r=[]
    for i in range(10, 46):
        t = t1.col_values(i, 1, rows+1)
        for j in range(0, rows):
            t[j] = float(t[j])
        r.append(t)
    # print r[35][82]
    # print x,'\n',y,'\n',r
    distMin,distMax,distSum,count=99999999,0,0,0
    for i in range(0, rows-1):
        for j in range(i+1, rows):
            dist = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            distSum=distSum+dist
            count=count+1
            if dist<distMin:
                distMin=dist
            if dist>distMax:
                distMax=dist
    print distMax,distMin,distSum/count
    hs=60000
    ht=1
    HS=6
    HT=1
    # print hs,ht
    gamma=[[] for i in range(0,HT)]
    for i in range(0,HT):
        gamma[i]=[j for j in range(0,HS)]
    # print len(gamma),len(gamma[0])
    distAve=[[] for i in range(0,HT)]
    for i in range(0,HT):
        distAve[i]=[j for j in range(0,HS)]
    npairsRec = [[] for i in range(0, HT)]
    for i in range(0, HT):
        npairsRec[i] = [j for j in range(0, HS)]
    # distAve,npairsRec=gamma,gamma

    for tt in range(0,HT):  #12   HT
        for ss in range(0,HS):
            snum = 0
            npairs = 0
            distSum=0
            for t1 in range(0, 36 - tt):#36 - tt
                t2 = t1 + ht*tt
                for i in range(0,rows):
                    if (r[t1][i] < 0):
                        continue
                    for j in range(0,rows):
                        if (r[t2][j] < 0):
                            continue
                        dist=math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
                        if (dist>(ss-1)*hs and dist<=ss*hs):
                            distSum = distSum + dist
                            snum=snum+(r[t1][i]-r[t2][j])**2
                            npairs=npairs+1
                            # print dist, (r[t1][i] - r[t2][j]) ** 2
                print t1, snum/npairs/2
                snum = 0
                npairs = 0
            # print ss,snum,npairs
            if (npairs==0):
                gamma[tt][ss] = 0
                npairsRec[tt][ss] = 0
                distAve[tt][ss] = 0
            else:
                gamma[tt][ss]=0.5*snum/npairs
                npairsRec[tt][ss] = npairs
                distAve[tt][ss]=distSum/npairs
    for tt in range(0,HT):
        print gamma[tt]
        # print distAve[tt]

    # N=HS
    # for tt in range(0,HT):
    #     b = np.mat(np.zeros((3, 1), np.float))
    #     c = np.mat(np.zeros((3, 1), np.float))
    #     # print b
    #     x = np.ones((N, 3), np.float)
    #     x[:, 1] = distAve[tt][0:N]
    #     for i in range(0,N):
    #         x[i, 2] = (distAve[tt][i]) ** 3
    #     x = np.mat(x)
    #     # print x
    #     # print distAve[tt][0],distAve[tt][1],distAve[tt][2]
    #     y = np.mat(np.array(gamma[tt][0:N], np.float)).T
    #     # print y
    #     b=(x.T*x).I*x.T*y
    #     # print b
    #     c[0]=b[0]
    #     c[1]=math.sqrt(-b[1]/b[2]/3)
    #     c[2]=2*c[1]*b[1]/3
    #     print c
    wb = xlwt.Workbook(encoding='utf-8')
    ws1 = wb.add_sheet('gammaValue')
    for tt in range(0,HT):
        for ss in range(0, HS):
            ws1.write(ss, tt, gamma[tt][ss])

    ws2 = wb.add_sheet('npairs')
    for tt in range(0, HT):
        for ss in range(0, HS):
            ws2.write(ss, tt, npairsRec[tt][ss])

    ws3 = wb.add_sheet('disAve')
    for tt in range(0, HT):
        for ss in range(0, HS):
            ws3.write(ss, tt, distAve[tt][ss])

    # out = path + "gammaValue_IMERG_10000_66_11.xls"  #
    # wb.save(out)

    print 'hahaha'

# Space_Time_Variogram()

def curveFitting():
    path = 'F:/Test/Paper180614/Result/'
    d = xlrd.open_workbook(path + 'gammaValueFitting.xlsx')
    print 'open file successfully!'
    HT=6
    hs=30000
    HS=22
    Spherical, Gaussian, Space, Time = 1, 1, 1, 0

    #
    t1 = d.sheet_by_index(3)  # 0是RGS，1是IMERG
    #
    rows, cols = t1.nrows, t1.ncols - 1
    print rows, cols
    dist = np.mat(np.ones((rows, 1), np.float))
    gamma = np.mat(np.ones((rows, cols), np.float))
    # print dist.shape,gamma.shape
    for i in range(0,rows):
        dist[i] = t1.cell_value(i, 0)
        for j in range(0,cols):
            gamma[i,j] = t1.cell_value(i,j+1)
    # distTime = np.mat(np.ones((1, cols), np.float))
    # for j in range(0, cols):
    #     distTime[j]=j
    distTime = [j for j in range(0,cols)]
    # print dist,'\n',dist.shape,'\n',gamma,'\n',gamma.shape
    if(Spherical and Space):
        method='Spherical'
        direction='Space'
        curveCoef=np.mat(np.ones((9,HT),np.float))
        for t in range(0,HT):
            Rsquare=0.5
            for i in range(4,HS):
                # print t,i,'///'
                y=gamma[0:i,t].copy()
                x=np.mat(np.ones((i,3),np.float))
                # print x.shape,y.shape
                for j in range(0,i):
                    x[j, 1] = dist[j]
                    x[j, 2] = dist[j] ** 3
                # print x
                tempCoef=curveCoef[:,0].copy()
                # print tempCoef.shape
                tempCoef[0:3,0]=(x.T*x).I*x.T*y
                if (tempCoef[1]>0 and tempCoef[2]>0 or tempCoef[1]<0 and tempCoef[2]<0):    #tempCoef[0]<0 or
                    # print 'error 1'
                    continue
                sse=(y-x*tempCoef[0:3]).T*(y-x*tempCoef[0:3])
                sst=(y-np.mean(y)).T*(y-np.mean(y))
                tempCoef[3]=1-(sse/sst) #R方
                tempCoef[4]=i
                tempCoef[5]=tempCoef[0]
                tempCoef[6]=np.sqrt(-tempCoef[1]/tempCoef[2]/3)
                tempCoef[7]=2*tempCoef[6]*tempCoef[1]/3
                if tempCoef[7]<=0:
                    # print 'error 2'
                    continue
                nn=int(tempCoef[6]/hs+2)
                if nn>rows:
                    nn=rows
                yobs=gamma[0:nn,t].copy()
                xobs=np.mat(np.ones((nn,3),np.float))
                # print x.shape,y.shape
                for j in range(0, nn-1):
                    xobs[j, 1] = dist[j]
                    xobs[j, 2] = dist[j] ** 3
                # print x
                ypre=xobs * tempCoef[0:3]
                if nn<=rows:
                    ypre[nn-1] = tempCoef[5]+tempCoef[7]
                sse = (yobs - ypre).T * (yobs - ypre)
                sst = (yobs - np.mean(yobs)).T * (yobs - np.mean(yobs))
                tempCoef[8] = 1 - (sse / sst)  # R方
                # print tempCoef
                if(tempCoef[8]>=Rsquare):
                    curveCoef[:, t]=tempCoef
                    Rsquare=tempCoef[8]
                else:
                    continue
            # print curveCoef[:, t]
        print curveCoef
    if(Spherical and Time):
        method = 'Spherical'
        direction = 'Time'
        curveCoef = np.mat(np.ones((9, rows), np.float))
        for s in range(0, 10):
            Rsquare = 0.8
            for i in range(4, cols):
                y = (gamma[s, 0:i].copy()).T
                x = np.mat(np.ones((i,3),np.float))
                # print x.shape,y.shape
                for j in range(0, i):
                    x[j, 1] = distTime[j]
                    x[j, 2] = distTime[j] ** 3
                # print x
                tempCoef = curveCoef[:, 0].copy()
                # print tempCoef.shape
                tempCoef[0:3, 0] = (x.T * x).I * x.T * y
                if (tempCoef[1] > 0 and tempCoef[2] > 0 or tempCoef[1] < 0 and tempCoef[2] < 0):
                    continue
                sse = (y - x * tempCoef[0:3]).T * (y - x * tempCoef[0:3])
                sst = (y - np.mean(y)).T * (y - np.mean(y))
                tempCoef[3] = 1 - (sse / sst)  # R方
                tempCoef[4] = i
                tempCoef[5] = tempCoef[0]
                tempCoef[6] = np.sqrt(-tempCoef[1] / tempCoef[2] / 3)
                tempCoef[7] = 2 * tempCoef[6] * tempCoef[1] / 3

                if (tempCoef[7] <= 0):
                    continue
                # print curveCoef[:, s]
                nn = int(tempCoef[6] / 1 + 2)
                print nn
                if nn>cols:
                    nn=cols
                yobs = (gamma[s,0:nn].copy()).T
                xobs = np.mat(np.ones((nn, 3), np.float))
                # print x.shape,y.shape
                for j in range(0, nn-1):
                    xobs[j, 1] = distTime[j]
                    xobs[j, 2] = distTime[j] ** 3
                # print x
                ypre = xobs * tempCoef[0:3]
                if nn<=cols:
                    ypre[nn - 1] = tempCoef[5] + tempCoef[7]
                sse = (yobs - ypre).T * (yobs - ypre)
                sst = (yobs - np.mean(yobs)).T * (yobs - np.mean(yobs))
                print sse,sst
                tempCoef[8] = 1 - (sse / sst)  # R方
                # print tempCoef
                if (tempCoef[8] >= Rsquare):
                    curveCoef[:, s] = tempCoef
                    Rsquare = tempCoef[8]
                else:
                    continue
            # print curveCoef[:, s]
        print curveCoef

    # print r[35][82]
    # print x,'\n',y,'\n',r
    distMin, distMax, distSum, count = 99999999, 0, 0, 0
    for i in range(0, rows - 1):
        for j in range(i + 1, rows):
            dist = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            distSum = distSum + dist
            count = count + 1
            if dist < distMin:
                distMin = dist
            if dist > distMax:
                distMax = dist
    print distMax, distMin, distSum / count
    hs = 60000
    ht = 1
    HS = 12
    HT = 6
    # print hs,ht
    gamma = [[] for i in range(0, HT)]
    for i in range(0, HT):
        gamma[i] = [j for j in range(0, HS)]
    # print len(gamma),len(gamma[0])
    distAve = [[] for i in range(0, HT)]
    for i in range(0, HT):
        distAve[i] = [j for j in range(0, HS)]
    npairsRec = [[] for i in range(0, HT)]
    for i in range(0, HT):
        npairsRec[i] = [j for j in range(0, HS)]
    # distAve,npairsRec=gamma,gamma

    for tt in range(0, HT):  # 12   HT
        for ss in range(0, HS):
            snum = 0
            npairs = 0
            distSum = 0
            for t1 in range(0, 36 - tt):  # 36 - tt
                t2 = t1 + ht * tt
                for i in range(0, rows):
                    if (r[t1][i] < 0):
                        continue
                    for j in range(0, rows):
                        if (r[t2][j] < 0):
                            continue
                        dist = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                        if (dist > (ss - 1) * hs and dist <= ss * hs):
                            distSum = distSum + dist
                            snum = snum + (r[t1][i] - r[t2][j]) ** 2
                            npairs = npairs + 1
                            # print dist, (r[t1][i] - r[t2][j]) ** 2
                # print t1, snum, npairs
            # print ss,snum,npairs
            if (npairs == 0):
                gamma[tt][ss] = 0
                npairsRec[tt][ss] = 0
                distAve[tt][ss] = 0
            else:
                gamma[tt][ss] = 0.5 * snum / npairs
                npairsRec[tt][ss] = npairs
                distAve[tt][ss] = distSum / npairs
    for tt in range(0, HT):
        print gamma[tt]
        # print distAve[tt]

    N = HS
    for tt in range(0, HT):
        b = np.mat(np.zeros((3, 1), np.float))
        c = np.mat(np.zeros((3, 1), np.float))
        # print b
        x = np.ones((N, 3), np.float)
        x[:, 1] = distAve[tt][0:N]
        for i in range(0, N):
            x[i, 2] = (distAve[tt][i]) ** 3
        x = np.mat(x)
        # print x
        # print distAve[tt][0],distAve[tt][1],distAve[tt][2]
        y = np.mat(np.array(gamma[tt][0:N], np.float)).T
        # print y
        b = (x.T * x).I * x.T * y
        # print b
        c[0] = b[0]
        c[1] = math.sqrt(-b[1] / b[2] / 3)
        c[2] = 2 * c[1] * b[1] / 3
        print c
    wb = xlwt.Workbook(encoding='utf-8')
    ws1 = wb.add_sheet('gammaValue')
    for tt in range(0, HT):
        for ss in range(0, HS):
            ws1.write(ss, tt, gamma[tt][ss])

    ws2 = wb.add_sheet('npairs')
    for tt in range(0, HT):
        for ss in range(0, HS):
            ws2.write(ss, tt, npairsRec[tt][ss])

    ws3 = wb.add_sheet('disAve')
    for tt in range(0, HT):
        for ss in range(0, HS):
            ws3.write(ss, tt, distAve[tt][ss])

    # out = path + "gammaValue_STL_RGS_30000_23_11.xls"  #
    # wb.save(out)

    print 'hahaha'


# curveFitting()

def StaTemp():
    path = 'F:/Test/Paper180614/Result/STVC/'
    d = xlrd.open_workbook(path + 'STVC.xls')
    print 'open file successfully!'
    t1 = d.sheet_by_index(0)  # 0是RGS，1是IMERG
    t2 = d.sheet_by_index(1)
    #
    if (t1.ncols!=t2.ncols or t1.nrows!=t2.nrows):
        print  'Error!!'
    rows, cols = t1.nrows, t1.ncols
    print rows, cols
    RGS = np.ones((rows, cols), np.float)
    IMERG= np.ones((rows, cols), np.float)
    for i in range(0,rows):
        for j in range(0,cols):
            RGS[i,j] = t1.cell_value(i,j)
            IMERG[i, j] = t2.cell_value(i, j)
    x=RGS.reshape(rows*cols,1)
    y=IMERG.reshape(rows*cols,1)
    xave=[sum(x) / len(x) for i in range(0,len(x))]
    yave=[sum(y) / len(y) for i in range(0,len(y))]
    xe= x-xave
    ye= y-yave

    Index=np.ones((5,1),np.float)
    Index[0,0]=sum(xe*ye)**2/(sum(xe**2)*sum(ye**2))
    Index[1, 0]=np.sqrt(sum((y-x)**2)/len(x))
    Index[2, 0]=sum(x)/sum(y)-1
    Index[3, 0]=sum(abs(x-y))/len(x)
    Index[4, 0]=sum((x-y)/(x+y)*2)/len(x)
    print Index
StaTemp()