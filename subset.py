# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
#####TRMM每天的降雨值#####
arcpy.env.workspace = "F:\\Research\\2006_2010\\day_TRMM\\201007\\tif"
days= arcpy.ListRasters()
_32="F:\\Research\\2006_2010\\hb_area\\32_32.shp"
for day in days:
    out=ExtractByMask(day,_32)
    out.save("F:\\Research\\2006_2010\\Subset_32\\day_32\\201007\\"+day[11:19]+"_32.tif")

arcpy.env.workspace = "F:\\Research\\2006_2010\\day_TRMM\\200812\\tif"
days= arcpy.ListRasters()
hb="F:\\Research\\2006_2010\\hb_area\\hb_clip.shp"
for day in days:
    out=ExtractByMask(day,hb)
    out.save("F:\\Research\\2006_2010\\Subset_hb\\day_hb\\200812\\"+day[11:19]+"_hb.tif")

# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
#####TRMM每天的降雨值#####
for i in range(6,11):
    arcpy.env.workspace = "F:\\Research\\2006_2010\\month_TRMM\\"+str(2000+i)
    months= arcpy.ListRasters()
    _32="F:\\Research\\2006_2010\\hb_area\\32_32.shp"
    for month in months:
        out=ExtractByMask(month,_32)
        out.save("F:\\Research\\2006_2010\\Subset_32\\month_32\\"+month[0:6]+"_32.tif")

    arcpy.env.workspace = "F:\\Research\\2006_2010\\month_TRMM\\"+str(2000+i)
    months= arcpy.ListRasters()
    hb="F:\\Research\\2006_2010\\hb_area\\hb_clip.shp"
    for month in months:
        out=ExtractByMask(month,hb)
        out.save("F:\\Research\\2006_2010\\Subset_hb\\month_hb\\"+month[0:6]+"_hb.tif")

# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
#####TRMM每天的降雨值#####
arcpy.env.workspace = "F:\\Research\\2006_2010\\year_TRMM"
years= arcpy.ListRasters()
_32="F:\\Research\\2006_2010\\hb_area\\32_32.shp"
for year in years:
    out=ExtractByMask(year,_32)
    out.save("F:\\Research\\2006_2010\\Subset_32\\year_32\\"+year[0:4]+"_32.tif")

arcpy.env.workspace = "F:\\Research\\2006_2010\\year_TRMM"
years= arcpy.ListRasters()
hb="F:\\Research\\2006_2010\\hb_area\\hb_clip.shp"
for year in years:
    out=ExtractByMask(year,hb)
    out.save("F:\\Research\\2006_2010\\Subset_hb\\year_hb\\"+year[0:4]+"_hb.tif")
