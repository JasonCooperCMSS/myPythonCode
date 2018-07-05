# coding:utf-8
import numpy as np
import math
import sympy
import arcpy
from arcpy import env
from arcpy.sa import *

def coarseGraining(field, coarseShape):
    # "计算聚合时的窗口大小"
    rowRatio = sympy.S(field.shape[0]) / coarseShape[0]
    colRatio = sympy.S(field.shape[1]) / coarseShape[1]
    # print rowRatio
    # "循环计算当前层级每个格网的取值"
    # "针对非整数倍的粗粒化（类似插值但不是），按占原整格面积的比例进行累加。"
    # "制作的window其实是一个面积比例，window与field相乘其实就只是按位置相乘"
    # "由于i,j是先行后列，所以我对h和v进行了调整，也先行后列，然后由于field[math.floor(bottom):math.ceil(top),math.floor(left):math.ceil(right)]是第一个参数为行，"
    # "后一个参数为列，所以两个参数的顺序也得调整"
    # "看似只不过减少了一行转置操作，实则在window的理解和field的循环上都更清楚了，先行后列！"
    coarseField = np.zeros((coarseShape), np.float)

    top = 0
    for i in range(0, coarseShape[0]):

        bottom = top
        top = bottom + rowRatio
        window_v = np.zeros((math.ceil(top) - math.floor(bottom)), np.float)
        for k in range(int(math.floor(bottom)), int(math.ceil(top))):
            if (k == math.floor(bottom)):
                window_v[k - math.floor(bottom)] = math.floor(bottom) + 1 - bottom
            elif (k == math.ceil(top) - 1):
                window_v[k - math.floor(bottom)] = top + 1 - math.ceil(top)
            else:
                window_v[k - math.floor(bottom)] = 1
        window_v.shape = len(window_v), 1
        # print(window_v)
        # print ("i=", i, "len(v)=", window_v.shape[0])
        right = 0
        for j in range(0, coarseShape[1]):
            left = right
            right = left + colRatio
            # print right
            window_h = np.zeros((math.ceil(right) - math.floor(left)), np.float)  #
            for k in range(int(math.floor(left)), int(math.ceil(right))):
                if (k == math.floor(left)):
                    window_h[k - math.floor(left)] = math.floor(left) + 1 - left
                elif (k == math.ceil(right) - 1):
                    window_h[k - math.floor(left)] = right + 1 - math.ceil(right)
                else:
                    window_h[k - math.floor(left)] = 1
            window_h.shape = 1, len(window_h)
            # print(window_h)
            # print ("j=", j,"len(h)=",window_h.shape[1])
            window = window_v * window_h
            # print window
            # window = np.transpose(window)
            # 对于数组的相乘，“*”号意思是对应相乘，对于矩阵来说才是矩阵相乘。
            coarseField[i, j] = np.sum(
                window * field[math.floor(bottom):math.ceil(top), math.floor(left):math.ceil(right)])
            # print(coarseField[i, j])
    coarseField = coarseField / (rowRatio * colRatio)
    return coarseField

from dbfread import DBF

# A="2Phigh"#ucPhighucPlow_supp
# dbf=DBF("F:\\Test\\GWR\\"+A+".dbf", load=True)
# a=[]
# for i in range(0,869632):
#     a.append(dbf.records[i]['Predicted'])#M201507/PredictedVARIABLE
# np.savetxt("F:/Test/OUT/"+"i"+A+".txt",a,fmt = '%.8f')

E="rMFnGWRnew"
A ="MFnGWRnew"
B ="d1"
D ="t1"
arcpy.env.workspace = "F:/Test/TEMP/"
arcpy.TableToDBASE_conversion('F:/Test/OUT/'+E+'.txt',"F:/Test/TEMP/")
arcpy.MakeTableView_management("F:/Test/TEMP/"+E+".dbf", B)

C="F:/Test/POINT/_869632.shp"
# C="F:/Test/POINT/_3397.shp"
arcpy.MakeFeatureLayer_management ( C,D )
arcpy.AddJoin_management(D, "ORIG_FID", B, "OID")

CELLSIZEX = arcpy.GetRasterProperties_management("F:/Test/TEMP/0/_01", "CELLSIZEX")
CELLSIZE = CELLSIZEX.getOutput(0)
CELLSIZE = str(float(CELLSIZE) /16)

arcpy.PointToRaster_conversion(D, E+".Field1",A+".tif", "MOST_FREQUENT", "NONE", CELLSIZE)
arcpy.RemoveJoin_management(D)
ExtractValuesToPoints("F:/Test/AN2/APoint.shp", A+".tif",A+'.shp',"INTERPOLATE","VALUE_ONLY")
arcpy.TableToExcel_conversion(A+'.shp', A+".xls")

# row=43
# col=79
# A="Plow"
# dbf=DBF("F:\\Test\\MLR\\"+A+".dbf", load=True)
# d=[]
#
# a=[]
# for i in range(0,row*col):
#     a.append(dbf.records[i]['POINT_X'])#M201507/
# d.append(a)
# # np.savetxt("F:/Test/OUT/"+"r1"+A+".txt",a,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r1"+A+'.txt',"F:/Test/OUT/")
# a=[]
# for i in range(0,row*col):
#     a.append(dbf.records[i]['POINT_Y'])
# d.append(a)
# # np.savetxt("F:/Test/OUT/"+"r2"+A+".txt",a,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r2"+A+'.txt',"F:/Test/OUT/")
#
# a=[]
# for i in range(0,row*col):
#     a.append(dbf.records[i]['DEM'])
# d.append(a)
# # np.savetxt("F:/Test/OUT/"+"r3"+A+".txt",a,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r3"+A+'.txt',"F:/Test/OUT/")
# a=[]
# for i in range(0,row*col):
#     a.append(dbf.records[i]['LTD'])
# d.append(a)
# # np.savetxt("F:/Test/OUT/"+"r4"+A+".txt",a,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r4"+A+'.txt',"F:/Test/OUT/")
# a=[]
# for i in range(0,row*col):
#     a.append(dbf.records[i]['LTN'])
# d.append(a)
# # np.savetxt("F:/Test/OUT/"+"r5"+A+".txt",a,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r5"+A+'.txt',"F:/Test/OUT/")
# a=[]
# for i in range(0,row*col):
#     a.append(dbf.records[i]['ucPlow'])
# d.append(a)
# # np.savetxt("F:/Test/OUT/"+"r6"+A+".txt",a,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r6"+A+'.txt',"F:/Test/OUT/")
# a=[]
# for i in range(0,row*col):
#     a.append(dbf.records[i]['cPlow'])
# d.append(a)
# # np.savetxt("F:/Test/OUT/"+"r7"+A+".txt",a,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r7"+A+'.txt',"F:/Test/OUT/")
# e=np.array(d).reshape(row*col,7)
# # np.transpose(e)
# np.savetxt("F:/Test/OUT/"+"r"+A+".txt",e,fmt = '%.8f')
# # arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r"+A1+'.txt',"F:/Test/OUT/")

