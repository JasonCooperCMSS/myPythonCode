# coding:utf-8
import arcpy
from dbfread import DBF
import xlrd
import xlwt
import matplotlib
import numpy as np
from scipy import stats
import sys, string, os

def STA(x=[],y=[]):
    #####R2，RMSE，MAE，Bias#####
    xave=np.mean(x)
    yave=np.mean(y)
    xsum=np.sum(x)
    ysum=np.sum(y)
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sum4 = 0.0
    sum5 = 0.0
    sum6 = 0.0
    sum7 = 0.0
    R = 0.0
    RMSE = 0.0
    MAE = 0.0
    BIAS = 0.0

    for i in range(0, len(x)):
        sum1 = sum1 + (x[i]-xave)*(y[i]-yave)
        sum2 = sum2 + (x[i]-xave)** 2
        sum3 = sum3 + (y[i]-yave)** 2
        sum4 = sum4 + (x[i]-y[i])** 2
        sum5 = sum5 + abs(x[i]-y[i])
    if (sum2 == 0.0 or sum3== 0.0 or sum4== 0.0 or ysum == 0.0):
        #   print("分母有0值")
        R[i] = 999999
        RMSE[i] = 999999  ###站点数量不同要适量调整###
        BIAS[i] = 999999
        MAE[i] = 999999  ###站点数量不同要适量调整###
    else:
        R= sum1 / np.sqrt(sum2 * sum3)
        RMSE= np.sqrt(sum4 / len(x))  ###站点数量不同要适量调整###
        BIAS = xsum / ysum- 1
        MAE = sum5/ len(x)  ###站点数量不同要适量调整###
        # print (str(R[i]))
    return [R,RMSE,BIAS,MAE]

#    for r in range(0, 83):
def MONTH():
    data = xlrd.open_workbook('F:/MSA/RAIN/STA/M/TRMMM.xls')
    rg = xlrd.open_workbook('F:/MSA/RAIN/STA/M/RG.xls')

    workbook = xlwt.Workbook(encoding='ascii')

    worksheet = workbook.add_sheet('44')
    tdata = data.sheet_by_index(0)
    trg = rg.sheet_by_index(0)
    worksheet.write(0, 0, '44')
    for c in range(0, 44):
        x=tdata.col_values(c,0,83)
        y=trg.col_values(c,0,83)
        z=STA(x,y)
        for i in range(0,4):
            worksheet.write(i+1, c, z[i])

#    worksheet1 = workbook.add_sheet('83')
    worksheet.write(5, 0, '83')
    for r in range(0,83):
        x=tdata.row_values(r,0,44)
        y=trg.row_values(r,0,44)
        z=STA(x,y)
        for i in range(0,4):
            worksheet.write(i+6, r, z[i])

    tdata1 = data.sheet_by_index(1)
    trg1 = rg.sheet_by_index(1)
    worksheet.write(10, 0, 's')
    for c in range(0, 1):
        x = tdata1.col_values(c, 0, 83)
        y = trg1.col_values(c, 0, 83)
        z = STA(x, y)
        for i in range(0, 4):
            worksheet.write(i+11, c, z[i])

    tdata2 = data.sheet_by_index(2)
    trg2 = rg.sheet_by_index(2)
    worksheet.write(15, 0, 't')
    for r in range(0,1):
        x=tdata2.row_values(r,0,44)
        y=trg2.row_values(r,0,44)
        z=STA(x,y)
        for i in range(0,4):
            worksheet.write(i+16, r, z[i])

    out = "F:/MSA/RAIN/STA/M/" + "test.xls"
    workbook.save(out)
# MONTH()

# def lky():
#     rain=[0 for x in range(0, 83)]
#     for x in range(0, 83):
#         rain[x] = [0 for y in range(0, 44)]
#     for r in range(0, 83):
#         for c in range(0, 44):
#             if (tdata.cell_value(r, c) == 0.0 and trg.cell_value(r, c) != 0.0):
#                 rain[r][c]=1
#             elif (tdata.cell_value(r, c) != 0.0 and trg.cell_value(r, c) == 0.0):
#                 rain[r][c] =2
#     x1=[0 for y in range(0, 83)]
#     x2 = [0 for y in range(0, 83)]
#     y1=[0 for y in range(0, 44)]
#     y2 = [0 for y in range(0, 83)]
#     for r in range(0, 83):
#         for c in range(0, 44):
#             if (rain[r][c]==1):
#                 x1[r]=x1[r]+1
#                 y1[c]=y1[c]+1
#             elif (rain[r][c] ==2):
#                 x2[r] = x2[r] + 1
#                 y2[c] = y2[c] + 1
#
#     worksheet.write(20, 0, 'slky')
#     for c in range(0,83):
#         worksheet.write(21, c, float(x1[c])/44)
#         worksheet.write(22, c, float(x2[c]) / 44)
#     worksheet.write(23, 0, 'tlky')
#     for r in range(0, 44):
#         worksheet.write(24, r, float(y1[r]) / 83)
#         worksheet.write(25, r, float(y2[r]) / 83)


def TRMM():
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('Sheet1')
    for y in range(14,15):
        for m in range(4,13):
            dbf=DBF("F:/SA/RG/TRMM/M/"+str(2000+y)+"/"+str((2000+y)*100+m) + "_RG.dbf",load=True)
            for r in range(0, 83):
                worksheet.write(r, m-4, dbf.records[r]['RASTERVALU'])
    for y in range(15,18):
        for m in range(1,13):
            dbf=DBF("F:/SA/RG/TRMM/M/"+str(2000+y)+"/"+str((2000+y)*100+m) + "_RG.dbf",load=True)
            for r in range(0, 83):
                worksheet.write(r, 9+(y-15)*12+m-1, dbf.records[r]['RASTERVALU'])
    out = "F:/SA/RAIN/RAIN/" + "test.xls"
    workbook.save(out)
# TRMM()


def IMERG():
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('Sheet1')

    x=[31,28,31,30,31,30,31,31,30,31,30,31]
    xx = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for y in range(14,15):
        for m in range(4,13):
            dbf=DBF("F:/MSA/RG/IMERG/M/"+str(2000+y)+"/"+str((2000+y)*100+m) + "_RG.dbf",load=True)
            for r in range(0, 83):
                worksheet.write(r, m-4, 24*x[m-1]*dbf.records[r]['RASTERVALU'])
    for y in range(15,16):
        for m in range(1,13):
            dbf=DBF("F:/MSA/RG/IMERG/M/"+str(2000+y)+"/"+str((2000+y)*100+m) + "_RG.dbf",load=True)
            for r in range(0, 83):
                worksheet.write(r, 9+(y-15)*12+m-1, 24*x[m-1]*dbf.records[r]['RASTERVALU'])
    for y in range(16,17):
        for m in range(1,13):
            dbf=DBF("F:/MSA/RG/IMERG/M/"+str(2000+y)+"/"+str((2000+y)*100+m) + "_RG.dbf",load=True)
            for r in range(0, 83):
                worksheet.write(r, 9+(y-15)*12+m-1, 24*xx[m-1]*dbf.records[r]['RASTERVALU'])
    for y in range(17,18):
        for m in range(1,13):
            dbf=DBF("F:/MSA/RG/IMERG/M/"+str(2000+y)+"/"+str((2000+y)*100+m) + "_RG.dbf",load=True)
            for r in range(0, 83):
                worksheet.write(r, 9+(y-15)*12+m-1, 24*x[m-1]*dbf.records[r]['RASTERVALU'])
    out = "F:/MSA/RAIN/" + "test.xls"
    workbook.save(out)
# IMERG()

def STE():
    workbook = xlwt.Workbook(encoding='ascii')

    worksheet = workbook.add_sheet('IMERGM')
    path="F:/MSA/RG/M/IMERGM/"
    files = os.listdir(path)
    dbf_file = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.dbf':
            dbf_file.append(path + files[i])
    for i in range(0, len(dbf_file)):
        if dbf_file[i][8:10]=='02':
            x=28
            if dbf_file[i][6:8]=='16':
                x=29
        elif dbf_file[i][8:10]=='04' or dbf_file[i][8:10]=='06' or dbf_file[i][8:10]=='09' or dbf_file[i][8:10]=='11':
            x=30
        else:
            x=31
        dbf = DBF(dbf_file[i], load=True)
        for r in range(0, 83):
            worksheet.write(r, i, 24 * x * dbf.records[r]['RASTERVALU'])

    worksheet = workbook.add_sheet('TRMMM')
    path = "F:/MSA/RG/M/TRMMM/"
    files = os.listdir(path)
    dbf_file = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.dbf':
            dbf_file.append(path + files[i])
    for i in range(0, len(dbf_file)):
        if dbf_file[i][8:10] == '02':
            x = 28
            if dbf_file[i][6:8] == '16':
                x = 29
        elif dbf_file[i][8:10] == '04' or dbf_file[i][8:10] == '06' or dbf_file[i][8:10] == '09' or dbf_file[i][ 8:10] == '11':
            x = 30
        else:
            x = 31
        dbf = DBF(dbf_file[i], load=True)
        for r in range(0, 83):
            worksheet.write(r, i, 24 * x * dbf.records[r]['RASTERVALU'])

    out = "F:/MSA/RAIN/" + "RAIN.xlsx"
    workbook.save(out)
#STE()

def ev():
    import arcpy
    from arcpy import env
    arcpy.CheckOutExtension("spatial")
    arcpy.gp.overwriteOutput = 1
    arcpy.env.workspace = "F:/MSA/DATA/TIF/TRMMM/"
    rasters = arcpy.ListRasters("*", "tif")
    mask = "F:\\SA\\TEMP\\RainGauges.shp"
    for raster in rasters:
        out1 = "F:/MSA/OUT/" + "MTRG" + raster[5:11] + ".shp"
        arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out1, "INTERPOLATE")
    print("All done")

    for i in range(14, 18):
        arcpy.env.workspace = "F:/SA/TRMM/M/" + str(2000 + i) + "/"
        rasters = arcpy.ListRasters("*", "tif")
        mask = "F:\\SA\\TEMP\\RainGauges.shp"
        arcpy.CreateFolder_management("F:/SA/OUT/", str(2000 + i))
        for raster in rasters:
            out1 = "F:/SA/OUT/" + str(2000 + i) + '/' + raster[5:11] + "_RG.shp"
            arcpy.gp.ExtractValuesToPoints_sa(mask, raster, out1)
    print("All done")