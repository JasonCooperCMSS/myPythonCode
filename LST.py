# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
##########
arcpy.env.workspace = "F:\\Research\\LST\\0712"
days= arcpy.ListRasters()
_area="F:\\Research\\LST\\area\\area.shp"
for day in days:
    out=ExtractByMask(day,_area)
    out.save("F:\\Research\\LST\\0712sub0\\"+day[8:16]+day[20:23]+"_34.tif")

import arcpy
from arcpy import env
from arcpy.sa import *
env.workspace = "F:\\Research\\LST\\0712sub0\\"
rasters=arcpy.ListRasters()
_34 = "F:\\Research\\LST\\area\\341000.tif"
for raster in rasters:
    outTimes = Times(raster,_34)
    outInt = Int(outTimes)
    outCon = Con(IsNull(outInt), 999999, outInt)
    nibbleOut = Nibble(outCon, outInt, "ALL_VALUES")
    outDivide = Divide(nibbleOut, _34)
    outDivide.save("F:\\Research\\LST\\0712nibble\\"+raster[0:11]+"_N.tif")


# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
##########
arcpy.env.workspace = "F:\\Research\\LST\\0712nibble"
days= arcpy.ListRasters()
for day in days:
    out="F:\\Research\\LST\\0712res\\"+day[0:11]+"_R.tif"
    arcpy.Resample_management(day, out, "0.0078125 0.0078125", "BILINEAR")
#    out.save("F:\\Research\\LST\\0712res\\"+day+"_R.tif")

# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
##########
arcpy.env.workspace = "F:\\Research\\LST\\0712res"
days= arcpy.ListRasters()
_32="F:\\Research\\2006_2010\\hb_area\\32_32.shp"
for day in days:
    out=ExtractByMask(day,_32)
    out.save("F:\\Research\\LST\\0712sub\\"+day[0:11]+"_32.tif")

