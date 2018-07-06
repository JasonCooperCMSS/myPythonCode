# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
from dbfread import DBF

# coding:utf-8
import numpy as np
import math
import sympy
import arcpy
from arcpy import env
from arcpy.sa import *


env.workspace = "F:/Test/data/ASTERDEM/"
path="F:/Test/data/ASTERDEM/DEMtif/"

# "拼接栅格"
def f1():
    arcpy.MosaicToNewRaster_management("F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N33E111_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N33E110_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N33E109_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N32E113_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N32E112_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N32E111_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N32E110_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N32E109_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E116_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E115_dem.ti;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E114_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E113_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E112_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E111_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E110_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N31E109_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E116_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E115_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E114_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E113_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E112_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E111_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E110_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E109_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N30E108_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E116_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E115_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E114_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E113_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E112_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E111_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E110_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E119_dem.tif;F:/Test/data/ASTERDEM/DEMtif/ASTGTM2_N29E108_dem.tif;",
                                   "F:/Test/data/ASTERDEM/",  "dem.tif","","16_BIT_SIGNED", "","1", "LAST","FIRST")


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
#43 79  688 1264    3397    869632  9480, 5160  1339, 729

#
def f2():
    # "栅格转数组，注意要上下翻转"
    inRaster = "F:/Test/data/LST/"+"LTD79N"+".tif"
    r = arcpy.Raster(inRaster)
    a = arcpy.RasterToNumPyArray(r)
    b=np.zeros((729,1339), np.float)
    for i in range(0,729):
        b[i,:]=a[728-i,:]

    # "聚合而不是重采样"
    c=coarseGraining(b, (688,1264))
    # "保存到txt"
    d=np.array(c).reshape(869632,1)

    A ="LTDhigh"
    np.savetxt('F:/Test/OUT/'+"r"+A+".txt", d,fmt = '%.8f')
    # "创建表图层、要素图层，属性相连"
    B ="d1"
    D ="t1"
    arcpy.env.workspace = "F:/Test/TEMP/"
    arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r"+A+'.txt',"F:/Test/TEMP/")
    arcpy.MakeTableView_management("r"+A+".dbf", B)
    C="F:/Test/POINT/_869632.shp"
    # C="F:/Test/POINT/_3397.shp"
    arcpy.MakeFeatureLayer_management ( C,D )
    arcpy.AddJoin_management(D, "ORIG_FID", B, "OID")
    # "获取像元大小，要素转栅格"
    CELLSIZEX = arcpy.GetRasterProperties_management("F:/Test/TEMP/0/_01", "CELLSIZEX")
    CELLSIZE = CELLSIZEX.getOutput(0)
    CELLSIZE = str(float(CELLSIZE) /16)
    arcpy.PointToRaster_conversion(D, "r"+A+".Field1",A+".tif", "MOST_FREQUENT", "NONE", CELLSIZE)
    # "移除连接"
    arcpy.RemoveJoin_management(D)

    print "ok!"

# inRaster = "F:/Test/TEMP/"+"cPhigh"+".tif"
# r = arcpy.Raster(inRaster)
# a = arcpy.RasterToNumPyArray(r)
# b=np.zeros((688,1264), np.float)
# for i in range(0,688):
#     b[i,:]=a[687-i,:]
#
# # c=coarseGraining(b, (43,79))
# d=np.array(b).reshape(869632,1)

# A ="GWR"
# np.savetxt('F:/Test/MF2/'+"r"+A+".txt", d,fmt = '%.8f')
#
# B ="d1"
# D ="t1"
# arcpy.env.workspace = "F:/Test/TEMP/"
# arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r"+A+'.txt',"F:/Test/TEMP/")
# arcpy.MakeTableView_management("r"+A+".dbf", B)
#
# # C="F:/Test/POINT/_869632.shp"
# C="F:/Test/POINT/_3397.shp"
# arcpy.MakeFeatureLayer_management ( C,D )
# arcpy.AddJoin_management(D, "ORIG_FID", B, "OID")
#
# CELLSIZEX = arcpy.GetRasterProperties_management("F:/Test/TEMP/0/_01", "CELLSIZEX")
# CELLSIZE = CELLSIZEX.getOutput(0)
# # CELLSIZE = str(float(CELLSIZE) /16)
#
# arcpy.PointToRaster_conversion(D, "r"+A+".Field1",A+".tif", "MOST_FREQUENT", "NONE", CELLSIZE)
# arcpy.RemoveJoin_management(D)
# print "ok!"

# import math
# print(math.floor(2),math.ceil(2.0))
# env.workspace = "F:/Test/data/LST/"
# raster="F:/Test/data/LST/"+"LTN79.tif"
# outTimes = Times(raster,1000)
# outInt = Int(outTimes)
# outCon = Con(IsNull(outInt), 999999, outInt)
# nibbleOut = Nibble(outCon, outInt, "ALL_VALUES")
# outFloat=Float(nibbleOut)
# outDivide = Divide(outFloat, 1000)
# outDivide.save("F:/Test/data/LST/"+"LTN79N.tif")

# outIDW = Idw("F:/Test/CAL2/CPIM.shp", "e", "F:/Test/CAL2/IM7943.tif","2")
# outIDW.save("F:/Test/CAL2/CIDW.tif")

# dbf=DBF("F:/Test/AN2/APCKriging.dbf", load=True)
# d=[]
# for i in range(0,30):
#     d.append(dbf.records[i]['RASTERVALU'])#M201507/
# print d

# A ="MFn"
# B ="d1"
# D ="t1"
# arcpy.env.workspace = "F:/Test/TEMP/"
# arcpy.TableToDBASE_conversion('F:/Test/OUT/'+"r"+A+'.txt',"F:/Test/TEMP/")
# arcpy.MakeTableView_management("r"+A+".dbf", B)
#
# C="F:/Test/POINT/_869632.shp"
# arcpy.MakeFeatureLayer_management ( C,D )
# arcpy.AddJoin_management(D, "ORIG_FID", B, "OID")
#
# CELLSIZEX = arcpy.GetRasterProperties_management("F:/Test/TEMP/0/_01", "CELLSIZEX")
# CELLSIZE = CELLSIZEX.getOutput(0)
# CELLSIZE = str(float(CELLSIZE) /16)
#
# arcpy.PointToRaster_conversion(D, "r"+A+".Field1",A+".tif", "MOST_FREQUENT", "NONE", CELLSIZE)
# arcpy.RemoveJoin_management(D)
#
# ExtractValuesToPoints("F:/Test/AN/APoint.shp", A+".tif",A+'.shp',"INTERPOLATE","VALUE_ONLY")
#
# arcpy.TableToExcel_conversion(A+'.shp', A+".xls")


