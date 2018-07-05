import arcpy
from arcpy import env
from arcpy.sa import *
import sys, string, os
for i in range(14,15):
        path = 'F:/SA/DAILY/TRMM/'+str(2000+i)+'/'
        files = os.listdir(path)
        Input_nc_file=[]
        for i in range(0,len(files)):
            if os.path.splitext(files[i])[1] == '.nc4':
                # Script arguments...
                Input_nc_file.append(path+files[i])
        #for i in range(0,len(files)):
        #    print Input_nc_file[i]
        #_area="F:\\SA\\TEMP\\1.shp"
        _area2="F:\\SA\\TEMP\\hb_clip.shp"
        for i in range(0,len(Input_nc_file)):#len(Input_nc_file)
        #    print Input_nc_file[i]
            out1 = "_1"#+Input_nc_file[i][35:43]
            arcpy.MakeNetCDFRasterLayer_md(Input_nc_file[i], "precipitation", "lon", "lat",out1)
        #    out2 = ExtractByMask(out1, _area)
            out3="_3"#+Input_nc_file[i][35:43]
            arcpy.Resample_management(out1, out3, "0.1", "BILINEAR")
            out4 = ExtractByMask(out3, _area2)
            out4.save("F:\\SA\\OUT\\" + "TRMM3B42D" +  Input_nc_file[i][33:41] + ".tif")


