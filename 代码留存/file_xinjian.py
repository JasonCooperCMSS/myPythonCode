import arcpy
from arcpy import env
for i in range(1,10):
    for j in range(10,13):
      arcpy.CreateFolder_management("F:\\Research\\2006_2010\\day_hbRG\\", "200"+str(i)+str(j)+"_RG")
print ("all done!")