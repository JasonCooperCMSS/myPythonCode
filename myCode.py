# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *

# "计算统计指标"
import numpy as np
def f3(x,y):
    xave=[sum(x) / float(len(x)) for i in range(0,len(x))]
    yave=[sum(y) / float(len(y)) for i in range(0,len(y))]
    xe=x-xave
    ye=y-yave

    R2=sum(xe*ye)/np.sqrt(sum(xe**2)*sum(ye**2))
    RMSE=np.sqrt(sum((x-y)**2)/float(len(x)))
    Bias=float(sum(x))/sum(y)-1
    MAE=sum(abs(x-y))/len(x)

    r = []
    r.append(R2)
    r.append(RMSE)
    r.append(Bias)
    r.append(MAE)
    return r

# "Excel文件读写"
def f5():
    import xlrd
    import xlwt
    data = "xxx.xlsx"
    excel = xlrd.open_workbook(data)
    table = excel.sheet_by_index(0)
    rows = table.nrows
    cols = table.ncols
    for i in range(0, rows):
        for j in range(1, cols):
            x=table.cell(i, j)
            xx=table.cell(i, j).value

    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('xx')
    for i in range(0, rows):
        for j in range(1, cols):
            worksheet.write(i, j, xx)

    out = "haha.xls"
    workbook.save(out)
# "系统文件路径获取"
def f4():
    import sys, string, os
    path = 'xxx/'
    files = os.listdir(path)
    Input_file = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.nc4':# "文件获取"
            Input_file.append(path + files[i])
        os.rename(path + '/' + files[i], path + '/' + 'xxx.txt')# "重命名"
# "栅格读成数组，数组转为栅格"
def f3():
    inRaster = "xxx.hdf"
    outRaster = "yyy.tif"
    r = arcpy.Raster(inRaster)  #"栅格对象""用Raster()函数声明栅格对象之后，就可以直接用+-*/等运算符完成地图代数了"
    print r
    a = arcpy.RasterToNumPyArray(r)  #"转数组"
    print len(a)
    ext = r.extent#"栅格范围"
    print ext.XMin, ext.YMin
    print r.meanCellWidth, r.meanCellWidth#"像元大小"
    llc = arcpy.Point(ext.XMin, ext.YMin)   #"点对象"
    r_scaled = arcpy.NumPyArrayToRaster((a), llc, r.meanCellWidth, r.meanCellWidth, (r.noDataValue))#"转栅格"
    r_scaled.save(outRaster)
# "最邻近插补"
def f2():

    env.workspace = "F:\\Research\\LST\\0712sub0\\"
    rasters = arcpy.ListRasters()
    _34 = "F:\\Research\\LST\\area\\341000.tif"
    for raster in rasters:
        outTimes = Times(raster, _34)
        outInt = Int(outTimes)
        outCon = Con(IsNull(outInt), 999999, outInt)
        nibbleOut = Nibble(outCon, outInt, "ALL_VALUES")
        outDivide = Divide(nibbleOut, _34)
        outDivide.save("F:\\Research\\LST\\0712nibble\\" + raster[0:11] + "_N.tif")
# "各类函数"
def f1():

    ##########
    arcpy.env.workspace = ""
    rasters = arcpy.ListRasters()
    mask=""
    for raster in rasters:
        out = ExtractByMask(raster, mask)   #"按掩膜提取"
        out.save("xxx/_34.tif")

        arcpy.Resample_management(raster, out, "xres yres", "BILINEAR")  # "NEAREST ","BILINEAR","CUBIC","MAJORITY" #"重采样"

        arcpy.ExtractSubDataset_management("xxx.hdf", "outfile.tif", "2")   #"提取子数据集，第三个参数是选择提取第几个子数据集（波段）"

        layer=""
        arcpy.MakeNetCDFRasterLayer_md(raster, "precipitation", "lon", "lat", layer)      # "nc制作图层"
        arcpy.CopyRaster_management(layer, out, format="TIFF")  # "图层保存为栅格"

        ExtractValuesToPoints(mask, raster,out, "INTERPOLATE","VALUE_ONLY") # "值提取到点"/"NONE","INTERPOLATE"/"VALUE_ONLY","ALL"

        out= SetNull(raster, raster, "Value=-3000") # "将满足条件的像元值设为Nodata"

        out=CellStatistics(rasters, out, "SUM", "NODATA")   # "像元统计" "MEAN/MAJORITY/MAXIMUM/MEDIAN/MINIMUM/MINORITY/RANGE/STD/SUM/VARIETY "    "NODATA"/"DATA"忽略nodata像元
        out.save("xxx.img")

        arcpy.Delete_management(raster) # "删除文件"

        rasters = arcpy.ListRasters()   # "数据的重命名"
        for raster in rasters:
            raster.save("xxx.tif")

        arcpy.TableToExcel_conversion(mask, "xxx.xls")# "表转Excel"

        arcpy.DirectionalDistribution_stats(raster, out, "1_STANDARD_DEVIATION", "xxx", "#")# "标准差椭圆"
        arcpy.MeanCenter_stats(raster, out, "xxx", "#", "#")    # "中心"

# ""
# ""
# ""
# ""
import sys, string, os
def CMORPH():

    # path = 'F:/Test/WuXia/CMORPH/nc/20090105/'
    # files = os.listdir(path)
    # Input_file = []
    # for i in range(0, len(files)):
    #     if os.path.splitext(files[i])[1] == '.nc':# "文件获取"
    #         Input_file.append(path + files[i])
    # for i in range(0, len(Input_file)):
    #     layer=str(20090105*100+i)
    #     arcpy.MakeNetCDFRasterLayer_md(Input_file[i], str(20090105*100+i), "lon", "lat", layer)  # "nc制作图层"
    #     out="F:/Test/WuXia/CMORPH/tif/20090105/"+str(20090105*100+i)+'.tif'
    #     arcpy.CopyRaster_management(layer, out, format="TIFF")  # "图层保存为栅格"

    path='F:/Test/WuXia/CMORPH/tif/20090105/'
    arcpy.env.workspace = path
    rasters = arcpy.ListRasters()

    out=CellStatistics(rasters,"SUM","NODATA")  # "像元统计" "MEAN/MAJORITY/MAXIMUM/MEDIAN/MINIMUM/MINORITY/RANGE/STD/SUM/VARIETY "    "NODATA"/"DATA"忽略nodata像元
    out.save("F:/Test/WuXia/CMORPH/tif/20090105.tif")
CMORPH()