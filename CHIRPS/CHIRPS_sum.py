# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
##########
for i in range(7,11):
    arcpy.env.workspace = "F:\\CHIRPS\\0data\\"+str(2000+i)+"\\"
    days= arcpy.ListRasters()
    _32="F:\\Research\\2006_2010\\hb_area\\32_32.shp"
    for day in days:
        out=ExtractByMask(day,_32)
        out.save("F:\\CHIRPS\\CHIRPS\\sub_32_C\\"+str(2000+i)+"\\"+day[12:16]+day[17:19]+day[20:22]+"_32.tif")

import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
for i in range(6,11):
    for j in range(1,13):
        arcpy.env.workspace = "F:\\CHIRPS\\CHIRPS\\sub_32_C\\"+str(2000+i)+"\\"+str((2000+i)*100+j)+"\\"
        rasters = arcpy.ListRasters("*", "tif")
        out = "F:\\CHIRPS\\CHIRPS\\month_C\\" +str(2000+i)+"\\"+ str((2000+i)*100+j) + ".tif"
        arcpy.gp.CellStatistics_sa((rasters), out, "SUM", "NODATA")

import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
for i in range(6,11):
    arcpy.env.workspace = "F:\\CHIRPS\\CHIRPS\\month_C\\"+str(2000+i)+"\\"
    rasters = arcpy.ListRasters("*", "tif")
    out = "F:\\CHIRPS\\CHIRPS\\year_C\\" +str(2000+i) + ".tif"
    arcpy.gp.CellStatistics_sa((rasters), out, "SUM", "NODATA")
