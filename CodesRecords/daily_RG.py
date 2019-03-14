import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
for i in range(6,7):
    for j in range(1,10):
        arcpy.env.workspace = "F:\\Research\\2006_2010\\day_TRMM\\"+"200"+str(i)+"0"+str(j)+"\\tif"
        rasters = arcpy.ListRasters("*", "tif")
        mask = "F:\\Research\\2006_2010\\hb_area\\hb_RG75.shp"
        for raster in rasters:
            out = "F:\\Research\\2006_2010\\day_hbRG\\"+"200"+str(i)+"0"+str(j)+"_RG\\" + raster[11:19]+"_RG.shp"
            arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out)
print("All done")


