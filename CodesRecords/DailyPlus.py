import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
for i in range(1,13):
    arcpy.env.workspace = "F:\\SA\\TRMM\\D\\"+str(i)+"\\" #自己修改
    rasters = arcpy.ListRasters("*", "tif")
    out= "F:\\SA\\TRMM\\M\\"+"TRMMM"+str(201500+i)+".tif"
    #for raster in rasters:
    arcpy.gp.CellStatistics_sa((rasters),out,"SUM","NODATA")
print("All done")

arcpy.env.workspace = "F:\\SA\\TRMM\\M\\" #自己修改
rasters = arcpy.ListRasters("*", "tif")
out= "F:\\SA\\TRMM\\Y\\"+"TRMMY2015"+".tif"
#for raster in rasters:
arcpy.gp.CellStatistics_sa((rasters),out,"SUM","NODATA")
print("All done")







