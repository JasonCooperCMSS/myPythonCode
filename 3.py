# coding:utf-8
import arcpy
from dbfread import DBF
import xlrd
import xlwt
def TRMM():
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('IMERG')
    worksheet1 = workbook.add_sheet('TRMM')
    worksheet2 = workbook.add_sheet('RG')
    for y in range(15,18):
            dbf=DBF("F:/MSA/RAIN/YE/"+"IYE"+str(2000+y)+ ".dbf",load=True)
            worksheet.write((y - 15), 0, dbf.records[0]['CenterX'])
            worksheet.write((y - 15), 1, dbf.records[0]['CenterY'])
            worksheet.write((y - 15) , 2 , dbf.records[0]['XStdDist'])
            worksheet.write((y - 15) , 3, dbf.records[0]['YStdDist'])
            worksheet.write((y - 15) , 4, dbf.records[0]['Rotation'])

            dbf1=DBF("F:/MSA/RAIN/YE/"+"TYE"+str(2000+y) + ".dbf",load=True)
            worksheet1.write((y - 15), 0, dbf1.records[0]['CenterX'])
            worksheet1.write((y - 15), 1, dbf1.records[0]['CenterY'])
            worksheet1.write((y - 15), 2 , dbf1.records[0]['XStdDist'])
            worksheet1.write((y - 15), 3, dbf1.records[0]['YStdDist'])
            worksheet1.write((y - 15), 4, dbf1.records[0]['Rotation'])

            dbf2=DBF("F:/MSA/RAIN/YE/"+"RYE"+str(2000+y)+ ".dbf",load=True)
            worksheet2.write((y - 15), 0, dbf2.records[0]['CenterX'])
            worksheet2.write((y - 15), 1, dbf2.records[0]['CenterY'])
            worksheet2.write((y - 15), 2 , dbf2.records[0]['XStdDist'])
            worksheet2.write((y - 15), 3, dbf2.records[0]['YStdDist'])
            worksheet2.write((y - 15), 4, dbf2.records[0]['Rotation'])
    out = "F:/MSA/RAIN/" + "test.xls"
    workbook.save(out)
# TRMM()
def bb():
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('IMERG')
    worksheet1 = workbook.add_sheet('TRMM')
    worksheet2 = workbook.add_sheet('RG')
    for y in range(14,15):
        for m in range(4,5):
            dbf=DBF("F:/MSA/RAIN/SE/"+"ISE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet.write((y - 14) * 4 + m - 4, 0, dbf.records[0]['CenterX'])
            worksheet.write((y - 14) * 4 + m - 4, 1, dbf.records[0]['CenterY'])
            worksheet.write((y - 14) * 4 + m - 4, 2 , dbf.records[0]['XStdDist'])
            worksheet.write((y - 14) * 4 + m - 4, 3, dbf.records[0]['YStdDist'])
            worksheet.write((y - 14) * 4 + m - 4, 4, dbf.records[0]['Rotation'])

            dbf1=DBF("F:/MSA/RAIN/SE/"+"TSE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet1.write((y - 14) * 4 + m - 4, 0, dbf1.records[0]['CenterX'])
            worksheet1.write((y - 14) * 4 + m - 4, 1, dbf1.records[0]['CenterY'])
            worksheet1.write((y - 14) * 4 + m - 4, 2 , dbf1.records[0]['XStdDist'])
            worksheet1.write((y - 14) * 4 + m - 4, 3, dbf1.records[0]['YStdDist'])
            worksheet1.write((y - 14) * 4 + m - 4, 4, dbf1.records[0]['Rotation'])

            dbf2=DBF("F:/MSA/RAIN/SE/"+"RSE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet2.write((y - 14) * 4 + m - 4, 0, dbf2.records[0]['CenterX'])
            worksheet2.write((y - 14) * 4 + m - 4, 1, dbf2.records[0]['CenterY'])
            worksheet2.write((y - 14) * 4 + m - 4, 2 , dbf2.records[0]['XStdDist'])
            worksheet2.write((y - 14) * 4 + m - 4, 3, dbf2.records[0]['YStdDist'])
            worksheet2.write((y - 14) * 4 + m - 4, 4, dbf2.records[0]['Rotation'])
    for y in range(15,17):
        for m in range(1,5):
            dbf=DBF("F:/MSA/RAIN/SE/"+"ISE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet.write((y - 14) * 4 + m - 4, 0, dbf.records[0]['CenterX'])
            worksheet.write((y - 14) * 4 + m - 4, 1, dbf.records[0]['CenterY'])
            worksheet.write((y - 14) * 4 + m - 4, 2 , dbf.records[0]['XStdDist'])
            worksheet.write((y - 14) * 4 + m - 4, 3, dbf.records[0]['YStdDist'])
            worksheet.write((y - 14) * 4 + m - 4, 4, dbf.records[0]['Rotation'])

            dbf1=DBF("F:/MSA/RAIN/SE/"+"TSE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet1.write((y - 14) * 4 + m - 4, 0, dbf1.records[0]['CenterX'])
            worksheet1.write((y - 14) * 4 + m - 4, 1, dbf1.records[0]['CenterY'])
            worksheet1.write((y - 14) * 4 + m - 4, 2 , dbf1.records[0]['XStdDist'])
            worksheet1.write((y - 14) * 4 + m - 4, 3, dbf1.records[0]['YStdDist'])
            worksheet1.write((y - 14) * 4 + m - 4, 4, dbf1.records[0]['Rotation'])

            dbf2=DBF("F:/MSA/RAIN/SE/"+"RSE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet2.write((y - 14) * 4 + m - 4, 0, dbf2.records[0]['CenterX'])
            worksheet2.write((y - 14) * 4 + m - 4, 1, dbf2.records[0]['CenterY'])
            worksheet2.write((y - 14) * 4 + m - 4, 2 , dbf2.records[0]['XStdDist'])
            worksheet2.write((y - 14) * 4 + m - 4, 3, dbf2.records[0]['YStdDist'])
            worksheet2.write((y - 14) * 4 + m - 4, 4, dbf2.records[0]['Rotation'])

    for y in range(17,18):
        for m in range(1,4):
            dbf=DBF("F:/MSA/RAIN/SE/"+"ISE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet.write((y - 14) * 4 + m - 4, 0, dbf.records[0]['CenterX'])
            worksheet.write((y - 14) * 4 + m - 4, 1, dbf.records[0]['CenterY'])
            worksheet.write((y - 14) * 4 + m - 4, 2 , dbf.records[0]['XStdDist'])
            worksheet.write((y - 14) * 4 + m - 4, 3, dbf.records[0]['YStdDist'])
            worksheet.write((y - 14) * 4 + m - 4, 4, dbf.records[0]['Rotation'])

            dbf1=DBF("F:/MSA/RAIN/SE/"+"TSE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet1.write((y - 14) * 4 + m - 4, 0, dbf1.records[0]['CenterX'])
            worksheet1.write((y - 14) * 4 + m - 4, 1, dbf1.records[0]['CenterY'])
            worksheet1.write((y - 14) * 4 + m - 4, 2 , dbf1.records[0]['XStdDist'])
            worksheet1.write((y - 14) * 4 + m - 4, 3, dbf1.records[0]['YStdDist'])
            worksheet1.write((y - 14) * 4 + m - 4, 4, dbf1.records[0]['Rotation'])

            dbf2=DBF("F:/MSA/RAIN/SE/"+"RSE"+str((2000+y)*100+m) + ".dbf",load=True)
            worksheet2.write((y - 14) * 4 + m - 4, 0, dbf2.records[0]['CenterX'])
            worksheet2.write((y - 14) * 4 + m - 4, 1, dbf2.records[0]['CenterY'])
            worksheet2.write((y - 14) * 4 + m - 4, 2 , dbf2.records[0]['XStdDist'])
            worksheet2.write((y - 14) * 4 + m - 4, 3, dbf2.records[0]['YStdDist'])
            worksheet2.write((y - 14) * 4 + m - 4, 4, dbf2.records[0]['Rotation'])
    out = "F:/MSA/RAIN/" + "test.xls"
    workbook.save(out)
def aa():
    #标准差椭圆
    input = "F:/MSA/RAIN/IRG.shp"
    for i in range(14, 15):
        for j in range(4, 13):
            output = "F:/MSA/OUT/" + "IME" + str((2000 + i) * 100 + j) + ".shp"
            arcpy.DirectionalDistribution_stats(input, output, "1_STANDARD_DEVIATION", "M" + str((2000 + i) * 100 + j),"#")
    #求重心
    input = "F:/MSA/RAIN/IRG.shp"
    for i in range(15, 15):
        for j in range(4, 5):
            output = "F:/MSA/RAIN/CENTERS/" + "RSE" + str(2000 + i) + str(j) + ".shp"
            arcpy.MeanCenter_stats(input, output, "S" + str(2000 + i) + str(j), "#", "#")
