# coding:utf-8
import arcpy
from arcpy import env
#####TRMM每天的降雨值#####
arcpy.env.workspace = "F:\\Research\\2006_2010\\year_TRMM"
months= arcpy.ListRasters()
_32="F:\\Research\\2006_2010\\hb_area\\32_32.shp"
for month in months:
    out="F:\\Research\\2006_2010\\Subset_32\\year_32\\"+month[0:5]+"_32.tif"
    arcpy.Clip_management(month,"#",out,_32,"NONE","ClippingGeometry","NO_MAINTAIN_EXTENT")

arcpy.env.workspace = "F:\\Research\\2006_2010\\year_TRMM"
months= arcpy.ListRasters()
hb="F:\\Research\\2006_2010\\hb_area\\hb_clip.shp"
for month in months:
    out="F:\\Research\\2006_2010\\Subset_hb\\year_hb\\"+month[0:5]+"_hb.tif"
    arcpy.Clip_management(month,"#",out,hb,"NONE","ClippingGeometry","NO_MAINTAIN_EXTENT")

