# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
import sys, string, os
from dbfread import DBF
import numpy as np
import xlrd
import xlwt


def nc2tif():
    # '月度数据转TIF,再合成年度'
    # for y in range(14,15):
    #     path = 'F:/SA/DATA/0/NCAREA1/' + 'IMERGM'+'/'+ 'M'+'/'+str(2000+y)+'/'
    #     files = os.listdir(path)
    #     Input_nc_file = []
    #     for j in range(0, len(files)):
    #         if os.path.splitext(files[j])[1] == '.nc' or os.path.splitext(files[j])[1] == '.nc4' :
    #             Input_nc_file.append(path + files[j])
    #
    #     arcpy.env.workspace = 'F:/SA/DATA/0/TIFAREA1/'+'IMERGM'+'/'+'M'+'/'+str(2000+y)+'/'
    #     for m in range(0,len(Input_nc_file)):#len(Input_nc_file)
    #     #    print Input_nc_file[i]
    #         layer=str((2000+y)*100+m+4)
    #         arcpy.MakeNetCDFRasterLayer_md(Input_nc_file[m], "precipitation", "lon", "lat",layer)
    #         r1 ='IM'+layer+'.tif'
    #         arcpy.CopyRaster_management(layer, r1, format="TIFF")
    #         print(m+4)

        # arcpy.env.workspace = 'F:/SA/DATA/0/TIFAREA1/' + 'IMERGM' + '/' + 'M' + '/' + str(2000 + y) + '/'
        # rasters = arcpy.ListRasters()
        # r2 = CellStatistics(rasters, "SUM", "DATA")
        # r2.save(
        #     'F:/SA/DATA/0/TIFAREA1/' + 'IMERGM' + '/' + 'Y' + '/'  + 'IM' + str(
        #         2000 + y)  + '.tif')
        # print(y)
    #     '转tif，日度合成月年'
    for y in range(16, 18):
        for m in range(1,13):
            path = 'F:/SA/DATA/0/NCAREA1/' + '3B42'+'/'+ 'D'+'/'+str(2000+y)+'/'+str(m)+'/'
            files = os.listdir(path)
            Input_nc_file = []
            for j in range(0, len(files)):
                if os.path.splitext(files[j])[1] == '.nc' or os.path.splitext(files[j])[1] == '.nc4' :
                    Input_nc_file.append(path + files[j])

            arcpy.env.workspace = 'F:/SA/DATA/0/TIFAREA1/'+'3B42'+'/'+'D'+'/'+str(2000+y)+'/'+str(m)+'/'
            for d in range(0,len(Input_nc_file)):#len(Input_nc_file)
            #    print Input_nc_file[i]
                layer=str((2000+y)*10000+(m*100)+d+1)
                arcpy.MakeNetCDFRasterLayer_md(Input_nc_file[d], "precipitation", "lon", "lat",layer)
                r1 ='T'+layer+'.tif'
                arcpy.CopyRaster_management(layer, r1, format="TIFF")
                print(d+1)

            arcpy.env.workspace = 'F:/SA/DATA/0/TIFAREA1/' + '3B42' + '/' + 'D' + '/' + str(2000 + y) + '/' + str(m) + '/'
            rasters = arcpy.ListRasters()
            r2 = CellStatistics(rasters, "SUM", "DATA")
            r2.save(
                'F:/SA/DATA/0/TIFAREA1/' + '3B42' + '/' + 'M' + '/' + str(2000 + y) + '/' + 'T' + str(
                    (2000 + y) * 100 + m) + '.tif')
            print(m)
        arcpy.env.workspace = 'F:/SA/DATA/0/TIFAREA1/' + '3B42' + '/' + 'M' + '/' + str(2000 + y) + '/'
        rasters = arcpy.ListRasters()
        r3 = CellStatistics(rasters, "SUM", "DATA")
        r3.save('F:/SA/DATA/0/TIFAREA1/' + '3B42' + '/' + 'Y' + '/'  + 'T' + str(2000 + y) + '.tif')
        print(2000+y)
#nc2tif()

# arcpy.env.workspace = 'F:/SA/RAIN/RAIN3/'
# RGS = 'F:/SA/TEMP/RGS.shp'
# arcpy.TableToExcel_conversion(RGS, "RGS.xls")

def f2():
    wb = xlwt.Workbook(encoding='ascii')
    ws1 = wb.add_sheet('IMERG')

    RGS = 'F:/SA/TEMP/RGS.shp'
    for y in range(15,18):
        arcpy.env.workspace = 'F:/SA/DATA/0/TIFAREA1/'+'IMERGD'+'/'+'M'+'/'+str(2000+y)+'/'
        rasters = arcpy.ListRasters()
        path = 'F:/SA/RG/' + 'IMERGD' + '/' + 'M' + '/' + str(2000 + y) + '/'

        for m in range(0,len(rasters)):
            shp=path+'I'+str((2000+y)*100+m+1)+'.shp'
            ExtractValuesToPoints(RGS, rasters[m],shp, "NONE", "VALUE_ONLY")
            xls=path+'I'+str((2000+y)*100+m+1)+'.xls'
            arcpy.TableToExcel_conversion(shp, xls)

            d=xlrd.open_workbook(xls)
            t=d.sheet_by_index(0)
            a=t.col_values(3, 1, 84)
            for i in range(0,83):
                ws1.write(i, (y-15)*12+m, a[i])

    ws2 = wb.add_sheet('IMERGM')
    for y in range(15, 18):
        arcpy.env.workspace = 'F:/SA/DATA/0/TIFAREA1/' + 'IMERGM' + '/' + 'M' + '/' + str(2000 + y) + '/'
        rasters = arcpy.ListRasters()
        path = 'F:/SA/RG/' + 'IMERGM' + '/' + 'M' + '/' + str(2000 + y) + '/'

        for m in range(0, len(rasters)):
            shp = path + 'IM' + str((2000 + y) * 100 + m + 1) + '.shp'
            print (rasters[m])
            ExtractValuesToPoints(RGS, rasters[m], shp, "NONE", "VALUE_ONLY")
            xls = path + 'IM' + str((2000 + y) * 100 + m + 1) + '.xls'
            arcpy.TableToExcel_conversion(shp, xls)
            print (xls)
            d2 = xlrd.open_workbook(xls)
            t2 = d2.sheet_by_index(0)
            a2 = t2.col_values(3, 1, 84)
            for i in range(0, 83):
                ws2.write(i, (y - 15) * 12 + m, a2[i])

    out = 'F:/SA/RAIN/RAIN3/' + "test.xls"
    wb.save(out)
#f2()
def f3(x,y):
    xave=[sum(x) / float(len(x)) for i in range(0,len(x))]
    yave=[sum(y) / float(len(y)) for i in range(0,len(y))]
    xe=x-xave
    ye=y-yave

    R2=sum(xe*ye)/np.sqrt(sum(xe**2)*sum(ye**2))
    RMSE=np.sqrt(sum((x-y)**2)/float(len(x)))
    Bias=float(sum(x))/sum(y)-1
    MAE=sum(abs(x-y))/len(x)

    r = []
    r.append(R2)
    r.append(RMSE)
    r.append(Bias)
    r.append(MAE)
    return r

def f4():
        data=xlrd.open_workbook('F:/SA/RAIN/RAIN3/CAL.xls')
        t1 = data.sheet_by_index(0)
        t2 = data.sheet_by_index(1)
        t3 = data.sheet_by_index(2)

        wb = xlwt.Workbook(encoding='ascii')
        ws1 = wb.add_sheet('I')
        ws2 = wb.add_sheet('IM')
        for i in range(1,37):
            a = t1.col_values(i, 1, 84)
            b = t2.col_values(i, 1, 84)
            c = t3.col_values(i, 1, 84)
            d = np.array(a)
            e = np.array(b)
            f = np.array(c)
            r1=f3(d,f)
            r2=f3(e,f)
            for j in range(0,4):
                ws1.write(j, i, r1[j])
                ws2.write(j, i, r2[j])
        out = 'F:/SA/RAIN/RAIN3/STA/' + "STA.xls"
        wb.save(out)
#f4()
def f5():
    data = xlrd.open_workbook('F:/SA/RAIN/RAIN3/ANA.xls')
    t1 = data.sheet_by_index(0)
    t2 = data.sheet_by_index(1)

    wb = xlwt.Workbook(encoding='ascii')
    ws1 = wb.add_sheet('I')
    for i in range(0, 12):
        a = t1.col_values(i, 1, 84)
        b = t2.col_values(i, 1, 84)
        d = np.array(a)
        e = np.array(b)
        r1 = f3(d, e)
        for j in range(0, 4):
            ws1.write(j, i, r1[j])

    for i in range(12, 16):
        a = t1.col_values(i, 1, 32)
        b = t2.col_values(i, 1, 32)
        d = np.array(a)
        e = np.array(b)
        r1 = f3(d, e)
        for j in range(0, 4):
            ws1.write(j+5, i, r1[j])

    for i in range(12, 16):
        a = t1.col_values(i, 32, 53)
        b = t2.col_values(i, 32, 53)
        d = np.array(a)
        e = np.array(b)
        r1 = f3(d, e)
        for j in range(0, 4):
            ws1.write(j+10, i, r1[j])

    for i in range(12, 16):
        a = t1.col_values(i, 53, 84)
        b = t2.col_values(i, 53, 84)
        d = np.array(a)
        e = np.array(b)
        r1 = f3(d, e)
        for j in range(0, 4):
            ws1.write(j + 15, i, r1[j])
    out = 'F:/SA/RAIN/RAIN3/STA/' + "STA.xls"
    wb.save(out)
#f5()

def f6():
    arcpy.env.workspace = r"F:/SA/RAIN/RAIN3/ELLIPSE/"
    I='I'
    R='R'
    RGS='RGS'
    arcpy.MakeTableView_management("F:/SA/RAIN/RAIN3/ELLIPSE/IMERG.dbf", I)
    arcpy.MakeTableView_management("F:/SA/RAIN/RAIN3/ELLIPSE/RGS.dbf", R)
    arcpy.MakeFeatureLayer_management("F:/SA/RAIN/RAIN3/ELLIPSE/RGSnew.shp", RGS)
    arcpy.AddJoin_management(RGS, "FID", I, "OID")
    arcpy.AddJoin_management(RGS, "FID", R, "OID")
    # for i in range(1,13):
        # out="F:/SA/RAIN/RAIN3/ELLIPSE/"+'I'+'CES'+str(i)+'.shp'
        # arcpy.DirectionalDistribution_stats(RGS, out, "1_STANDARD_DEVIATION", "IMERG.S"+str(i), "#")
        # out = "F:/SA/RAIN/RAIN3/ELLIPSE/" + 'I' + 'CES' + str(i) + '.shp'
        # arcpy.MeanCenter_stats(RGS, out, "IMERG.S"+str(i), "#", "#")

    for i in range(1, 13):
        # out = "F:/SA/RAIN/RAIN3/ELLIPSE/" + 'T' + 'ES' + str(i) + '.shp'
        # arcpy.DirectionalDistribution_stats(RGS, out, "1_STANDARD_DEVIATION","RGS.S" + str(i), "#")
        out = "F:/SA/RAIN/RAIN3/ELLIPSE/" + 'T' + 'CES' + str(i) + '.shp'
        arcpy.MeanCenter_stats(RGS, out, "RGS.S"+str(i), "#", "#")

#f6()
def f7():
    for i in range(1,13):
        input_features = "F:/SA/RAIN/RAIN3/ELLIPSE/" + 'I' + 'CES' + str(i) + '.shp'
        # output data
        output_feature_class = r"F:/SA/RAIN/RAIN3/ELLIPSE/CC/" + 'I' + 'CES' + str(i) + '.shp'
        # create a spatial reference object for the output coordinate system
        out_coordinate_system = arcpy.SpatialReference(32649)
        # run the tool
        arcpy.Project_management(input_features, output_feature_class, out_coordinate_system)
        layer='I' + 'CES' + str(i) + '.shp'
        arcpy.MakeFeatureLayer_management(output_feature_class, layer)
        arcpy.AddXY_management(layer)

    for i in range(1, 13):
        input_features = "F:/SA/RAIN/RAIN3/ELLIPSE/" + 'T' + 'CES' + str(i) + '.shp'
        # output data
        output_feature_class = r"F:/SA/RAIN/RAIN3/ELLIPSE/CC/" + 'T' + 'CES' + str(i) + '.shp'
        # create a spatial reference object for the output coordinate system
        out_coordinate_system = arcpy.SpatialReference(32649)
        # run the tool
        arcpy.Project_management(input_features, output_feature_class, out_coordinate_system)
        layer='T' + 'CES' + str(i) + '.shp'
        arcpy.MakeFeatureLayer_management(output_feature_class, layer)
        arcpy.AddXY_management(layer)
#f7()

def f8():
    data = xlrd.open_workbook('F:/WORK1/RAIN/3B43Y.xlsx')
    t1 = data.sheet_by_index(0)
    t2 = data.sheet_by_index(1)

    wb = xlwt.Workbook(encoding='ascii')
    ws1 = wb.add_sheet('TRY')
    for i in range(0, 1):
        a = t1.col_values(i, 0, 75)
        b = t2.col_values(i, 0, 75)
        d = np.array(a)
        e = np.array(b)
        r1 = f3(d, e)
        for j in range(0, 4):
            ws1.write(j, i, r1[j])
    out = 'F:/WORK1/RAIN/' + "STA4.xls"
    wb.save(out)
f8()

def nc22tif():
    #     '转tif，日度合成月年'
    for y in range(98, 116):
        arcpy.env.workspace = 'F:/WORK1/DATA/1998_2016TRMM/'+ str(1900 + y) + '/'
        rasters = arcpy.ListRasters()
        r3 = CellStatistics(rasters, "SUM", "DATA")
        r3.save('F:/WORK1/DATA/Y/' +str(1900 + y) + '.tif')
        print(1900+y)
    arcpy.env.workspace = 'F:/WORK1/DATA/Y/'
    rasters = arcpy.ListRasters()
    r3 = CellStatistics(rasters, "SUM", "DATA")
    r3.save('F:/WORK1/DATA/'  + '1998_2015.tif')
    print("1998_2015")

#nc22tif()