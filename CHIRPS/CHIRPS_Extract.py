import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
arcpy.env.workspace = "F:\\CHIRPS\\CHIRPS\\day_C\\2010\\201007\\"
rasters = arcpy.ListRasters("*", "tif")
mask = "F:\\CHIRPS\\hb_area\\hb_RG75.shp"
for raster in rasters:
    out = "F:\\CHIRPS\\CHIRPS\\day_hbRG\\201007\\"+raster[0:8]+ "_RG.shp"
    arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out)

import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
for i in range(6,11):
    arcpy.env.workspace = "F:\\CHIRPS\\CHIRPS\\month_C\\"+str(2000+i)+"\\"
    rasters = arcpy.ListRasters("*", "tif")
    mask = "F:\\CHIRPS\\hb_area\\hb_RG75.shp"
    for raster in rasters:
        out = "F:\\CHIRPS\\CHIRPS\\month_hbRG\\"+str(2000+i)+"\\"+raster[0:6]+ "_RG.shp"
        arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out)

import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
arcpy.env.workspace = "F:\\CHIRPS\\CHIRPS\\year_C\\"
rasters = arcpy.ListRasters("*", "tif")
mask = "F:\\CHIRPS\\hb_area\\hb_RG75.shp"
for raster in rasters:
    out = "F:\\CHIRPS\\CHIRPS\\year_hbRG\\"+raster[0:4]+ "_RG.shp"
    arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out)