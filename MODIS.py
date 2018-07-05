import os
import sys
import arcpy

# in_raster = "F:/Test/data/LST/MODISHDF/"+"MOD11A1.A2016183.h26v05.006.2016241045859.hdf"
# scale_factor ="0.02"
# out_raster = "F:/Test/data/LST/"+"183.tif"



# inRaster = "F:/Test/data/LST/MODISHDF/"+"MOD11A1.A2016183.h26v05.006.2016241045859.hdf"
# scaleFactor = 0.02
# outRaster = "F:/Test/data/LST/"+"2605.tif"

arcpy.ExtractSubDataset_management(inRaster,"F:/Test/data/LST/"+"A2605.tif")


# r = arcpy.Raster(inRaster)
# print r
# a = arcpy.RasterToNumPyArray(r)
# for i in range(0,len(a)):
#     print a[0]
# ext = r.extent
# print ext.XMin, ext.YMin
# print r.meanCellWidth, r.meanCellWidth
# llc = arcpy.Point(ext.XMin, ext.YMin)
# r_scaled = arcpy.NumPyArrayToRaster((a), llc, r.meanCellWidth, r.meanCellWidth, (r.noDataValue*scaleFactor))
# r_scaled.save(outRaster)

