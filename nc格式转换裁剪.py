import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
arcpy.env.workspace = "C:\\Users\\xia\\Desktop\\2006_2010\\200605\\hbclip"
rasters = arcpy.ListRasters("*", "tif")
mask= "C:\\Users\\xia\\Desktop\\2006_2010\\201007RG\\201007RG75.shp"
for raster in rasters:
    print(raster)
    out= "C:\\Users\\xia\\Desktop\\2006_2010\\hbpoint_RG\\"+raster[0:9]+"RG"+".shp"
    arcpy.gp.ExtractValuesToPoints_sa(mask,raster,out)
    print(raster[0:9]+"RG"+ ".shp"+"  has done")
print("All done")


