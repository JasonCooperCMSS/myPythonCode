import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1

for i in range(16,18):
        arcpy.env.workspace = 'F:/SA/TRMM/M/'+str(2000+i) +'/' # 自己修改
        rasters = arcpy.ListRasters("*", "tif")
        out = 'F:/SA/TRMM/Y/'+"TRMMY"+str(2000+i) +".tif" # 自己修改
         # for raster in rasters:
        arcpy.gp.CellStatistics_sa((rasters), out, "SUM", "NODATA")
print("All done")

for i in range(14,15):
    for j in range(4,13):
        arcpy.env.workspace = 'F:/SA/TRMM/D/'+str(2000+i) +'/'+str(j)+'/' # 自己修改
        rasters = arcpy.ListRasters("*", "tif")
        out = 'F:/SA/TRMM/M/'+str(2000+i) +'/'+"TRMMM"+str((2000+i)*100+j) +".tif" # 自己修改
         # for raster in rasters:
        arcpy.gp.CellStatistics_sa((rasters), out, "SUM", "NODATA")
print("All done")


