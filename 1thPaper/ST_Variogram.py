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

    path = 'F:/Test/Paper180614/Result/Temp/'
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
    # print r[0][0]
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

    #hs为空间步长，ht为时间步长，HS、HT为对应的要计算的范围，如HS=12则把距离在660000m以内的点对数都计算上
    hs=60000
    ht=1
    HS=12
    HT=12
    # print hs,ht
    gamma=[[] for i in range(0,HT)]
    distAve = [[] for i in range(0, HT)]
    npairsRec = [[] for i in range(0, HT)]
    for i in range(0,HT):
        gamma[i]=[j for j in range(0,HS)]
        distAve[i] = [j for j in range(0, HS)]
        npairsRec[i] = [j for j in range(0, HS)]
    # print len(gamma),len(npairsRec[0])

    for tt in range(0,HT):  #12   HT
        for ss in range(0,HS): #12
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
                # print t1, snum/npairs/2
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

    wb = xlwt.Workbook(encoding='utf-8')
    ws1 = wb.add_sheet('gammaValue')
    ws2 = wb.add_sheet('npairs')
    ws3 = wb.add_sheet('disAve')
    for tt in range(0,HT):
        for ss in range(0, HS):
            ws1.write(ss, tt, gamma[tt][ss])
            ws2.write(ss, tt, npairsRec[tt][ss])
            ws3.write(ss, tt, distAve[tt][ss])

    out = path + "gammaValue_RGS_60000_11_11.xls"  #
    wb.save(out)
    print 'Well done!'
Space_Time_Variogram()
