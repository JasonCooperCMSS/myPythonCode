import arcpy
from arcpy import env
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
arcpy.env.workspace = "F:/SA/TRMM/Y/"
rasters = arcpy.ListRasters("*", "tif")
mask = "F:\\SA\\TEMP\\RainGauges.shp"
arcpy.CreateFolder_management("F:/SA/OUT/", "Y")
for raster in rasters:
    out1 = "F:/SA/OUT/Y/" + raster[5:9] + "_RG.shp"
    arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out1)
print("All done")

for i in range(14, 18):
    arcpy.env.workspace = "F:/SA/TRMM/M/"+str(2000+i)+"/"
    rasters = arcpy.ListRasters("*", "tif")
    mask = "F:\\SA\\TEMP\\RainGauges.shp"
    arcpy.CreateFolder_management("F:/SA/OUT/", str(2000+i))
    for raster in rasters:
        out1 = "F:/SA/OUT/" +str(2000+i)+'/'+ raster[5:11] + "_RG.shp"
        arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out1)
print("All done")





for i in range(1,13):
    arcpy.env.workspace = "F:/SA/IMERG/D/"+str(i)+"/"
    rasters = arcpy.ListRasters("*", "tif")
    mask = "F:\\Research\\2006_2010\\hb_area\\hb_RG75.shp"
    arcpy.CreateFolder_management("F:/SA/OUT/", str(i))
    for raster in rasters:
        out1 = "F:/SA/OUT/"+str(i)+"/"+raster[7:15]+"_RG.shp"
        arcpy.gp.ExtractValuesToPoints_sa(mask,raster,out1)
print("All done")






arcpy.env.workspace = "F:/SA/MONTH/IMERGM/1/"
rasters = arcpy.ListRasters("*", "tif")
mask = ""
arcpy.CreateFolder_management("F:/SA/OUT/", "MI")
for raster in rasters:
    out1 = "F:/SA/OUT/MI/" + raster[6:12] + "_RG.shp"
    arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out1)
print("All done")

arcpy.env.workspace = "F:/SA/MONTH/3B43M/1/"
rasters = arcpy.ListRasters("*", "tif")
mask = ""
arcpy.CreateFolder_management("F:/SA/OUT/", "M3")
for raster in rasters:
    out1 = "F:/SA/OUT/M3/" + raster[5:11] + "_RG.shp"
    arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out1)
print("All done")