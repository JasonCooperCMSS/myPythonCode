#批量excel转表
import arcpy
arcpy.env.workspace = "F:\Research\RainGauge\RAIN"
for i in range(1,13):
    arcpy.ExcelToTable_conversion("F:\\Research\\RainGauge\\RAIN\\"+"2010年"+str(i)+"月累积降雨量.xls", "F:\\Research\\RainGauge\\RG_dbf\\"+str(2010*100+i)+".dbf", 'Sheet1')
print "all done!"

#删除字段
import arcpy
for i in range(6,11):
    for j in range(8,13):
        arcpy.env.workspace = "F:\\Research\\2006_2010\\day_hbRG\\"+str((2000+i)*100+j)+"_RG\\"
        shps=arcpy.ListFeatureClasses()
        for shp in shps:
            arcpy.DeleteField_management(shp,["站点名"])
print ("all done!")

#批量新建文件夹
import arcpy
from arcpy import env
for i in range(1,11):
    for j in range(1,13):
      arcpy.CreateFolder_management("F:\\Research\\RainGauge\\Statistics\\daily\\",str((2000+i)*100+j )+"_sta")
print ("all done!")

#批量提取到点
import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
arcpy.env.workspace = "F:\\Research\\2006_2010\\day_TRMM\\200812\\tif\\"
rasters = arcpy.ListRasters("*", "tif")
mask= "F:\\Research\\2006_2010\\08121001\\0812hb_RG75.shp"
for raster in rasters:
#    print(raster)
    out= "F:\\Research\\2006_2010\\day_hbRG\\200812_RG\\"+raster[11:19]+"_RG"+".shp"
    arcpy.gp.ExtractValuesToPoints_sa(mask,raster,out)
print("All done")

#重命名
import arcpy
arcpy.CheckOutExtension("spatial")
arcpy.gp.overwriteOutput=1
arcpy.env.workspace = 'F:/SA/TRMM/M/2017/'
rasters = arcpy.ListRasters("*", "tif")
for raster in rasters:
    arcpy.Rename_management(raster,'TRMMM'+raster[5:11]+ ".tif")