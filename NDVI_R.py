# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
##########
arcpy.env.workspace = "F:\\Research\\NDVI\\0data"
days= arcpy.ListRasters()
_area="F:\\Research\\NDVI\\area\\area.shp"
for day in days:
    out=ExtractByMask(day,_area)
    out.save("F:\\Research\\NDVI\\sub0\\"+day[0:6]+"_34.tif")

import arcpy
from arcpy import env
from arcpy.sa import *
env.workspace = "F:\\Research\\NDVI\\sub0"
rasters=arcpy.ListRasters()
for raster in rasters:
    outSetnull=SetNull(raster,raster,"Value=-3000")
    nibbleOut = Nibble(raster,outSetnull,  "ALL_VALUES")
    nibbleOut.save("F:\\Research\\NDVI\\nibble\\"+raster[0:6]+"_N.tif")

# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
##########
arcpy.env.workspace = "F:\\Research\\NDVI\\nibble"
days= arcpy.ListRasters()
_area="F:\\Research\\2006_2010\\hb_area\\hb_clip.shp"
for day in days:
    out=ExtractByMask(day,_area)
    out.save("F:\\Research\\NDVI\\NDVI_sub\\"+day[0:6]+"_hb.tif")