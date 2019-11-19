# coding:utf-8
import arcpy,os
from arcpy import env,sa
from arcpy.sa import *
arcpy.gp.overwriteOutput=1

path="F:\\Test\\Temp\\"
ras=path+"I201502.tif"
shpIn=path+"HB_CITIES.shp"

pathOut="F:\\Test\\Temp\\Out\\"

cursor=arcpy.SearchCursor(shpIn)
dicClass=[]
field='ID_1'
for row in cursor:
    t=row.getValue(field)
    if t not in dicClass:
        dicClass.append(t)
print dicClass
shpClass="shpClass"
for cls in dicClass:
    pathClass = pathOut+'Class/'
    if os.path.exists(pathClass) == False:
        os.makedirs(pathClass)
    shpClass =pathClass+ str(cls) + '.shp'
    where_clause = field+'= '+str(cls)
    arcpy.Select_analysis(shpIn, shpClass,where_clause)

    arcpy.AddField_management(shpClass, "_ID", "STRING")
    arcpy.CalculateField_management(shpClass, "_ID",
                                    '!FID!', "PYTHON_9.3")

    pathCls=pathOut+'Class_'+str(cls)+'/'
    if os.path.exists(pathCls) == False:
        os.makedirs(pathCls)
    arcpy.env.workspace = pathCls
    arcpy.Split_analysis(shpClass, shpClass, "_ID", pathCls)
    shpsCls=arcpy.ListFeatureClasses()
    num=0
    pathClsBod = pathOut + 'ClsBod_' + str(cls) + '/'
    if os.path.exists(pathClsBod) == False:
        os.makedirs(pathClsBod)
    for shp in shpsCls:
        print shp
        shpBound=pathClsBod+"Bound_Cls"+str(num)+'.shp'
        arcpy.MinimumBoundingGeometry_management(shp,shpBound,"RECTANGLE_BY_AREA", "NONE")
        rasOut=ExtractByMask(ras,shpBound)
        pathExtr = pathOut + str(cls)+'_Extr/'
        if os.path.exists(pathExtr) == False:
            os.makedirs(pathExtr)
        rasOut.save(pathExtr+str(num)+'.tif')
        num+=1