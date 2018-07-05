#生成表格标题
# coding:utf-8
import xlrd
import xlwt

#文件重命名，os.rename()
def jpg():
    path = 'C:/Users/Jason/Desktop/JPG/'
    files= os.listdir(path)
    for i in range(0,len(files)):
        os.rename(path+'/'+files[i], path+'/'+files[i])

#写入excel
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('monthlabel')
for x in range(6,18):
    for y in range(1,13):
        worksheet.write(0, (x-6)*12+y-1,"M"+str((2000+x)*100+y) )
out="F:/SA/RAIN/RAIN/"+"monthlabel.xls"
workbook.save(out)

import arcpy
arcpy.env.workspace = "C:/Users/Jason/Documents/ArcGIS/Default.gdb/"
rasters = arcpy.ListRasters("_3*", "ALL")
for raster in rasters:
    arcpy.Delete_management(raster)
print ("All done")

import arcpy
from arcpy import env
from arcpy.sa import *
import sys, string, os
path = 'F:\\SA\\IMERG\\'
files = os.listdir(path)
Input_nc_file=[]
for i in range(0,len(files)):
    if os.path.splitext(files[i])[1] == '.nc':
        # Script arguments...
        Input_nc_file.append(path+files[i])
# print Input_nc_file
_area="F:\\SA\\TEMP\\1.shp"
_area2="F:\\SA\\TEMP\\hb_clip.shp"
for i in range(0,len(Input_nc_file)):#
    print Input_nc_file[i]
    out1 = "_1"+Input_nc_file[i][35:43]
    arcpy.MakeNetCDFRasterLayer_md(Input_nc_file[i], "precipitationCal", "lon", "lat",out1)
    out2 = ExtractByMask(out1, _area)
    out3="_3"+Input_nc_file[i][35:43]
    arcpy.Resample_management(out2, out3, "0.1", "BILINEAR")
    out4 = ExtractByMask(out3, _area2)
    out4.save("F:\\SA\\OUT\\" + "IMERGDF" +  Input_nc_file[i][35:43] + ".tif")

import arcpy
from arcpy import env
from arcpy.sa import *
_area="F:\\Research\\NDVI\\area\\area.shp"
for i in range(1,13):
    file="C:\\Users\\Jason\\Desktop\\TRMM2013\\nc\\"+"3B43."+str(2013*100+i)+"01.7.HDF"+".nc"
    out = "3B43." + str(2013*100+i )
    arcpy.MakeNetCDFRasterLayer_md(file, "precipitation", "nlon", "nlat",out)
    trmm = ExtractByMask(out, _area)
    trmm.save("C:\\Users\\Jason\\Desktop\\TRMM2013\\OUT\\" + out + ".tif")

# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
arcpy.env.workspace = "C:\\Users\\Jason\\Desktop\\TRMM2013\\HDF"
rasters = arcpy.ListRasters()
for raster in rasters:
    raster.save("C:\\Users\\Jason\\Desktop\\TRMM2013\\OUT\\" + raster[0:14]+ ".tif")


# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
env.workspace = "F:\\Research\\LST\\LTN_5\\_32\\2010\\"
rasters = arcpy.ListRasters("*", "tif")
out= CellStatistics(rasters , "MEAN", "DATA")
out.save("F:\\Research\\LST\\LTNY\\_32\\"+"2010"+"LTN_32.tif" )


import arcpy
from arcpy import env
from arcpy.sa import *
env.workspace = "F:\\Research\\LST\\LTNsub0\\"
rasters=arcpy.ListRasters()
_34 = "F:\\Research\\LST\\area\\341000.tif"
for raster in rasters:
    outTimes = Times(raster,_34)
    outInt = Int(outTimes)
    outCon = Con(IsNull(outInt), 999999, outInt)
    nibbleOut = Nibble(outCon, outInt, "ALL_VALUES")
    outDivide = Divide(nibbleOut, _34)
    outDivide.save("F:\\Research\\LST\\LTNnibble\\"+raster[0:11]+"_N.tif")



