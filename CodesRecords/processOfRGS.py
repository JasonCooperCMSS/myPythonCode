# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
import sympy

import sys, string, os
import xlrd
import xlwt

#   将2011年1月到2017年12月间站点降水数据合并到一个xls文件，某些月份有重复行
def f1():
    path = 'F:/Test/Data/RGS/RGS83_201101_201712/'
    filesPath=path+'month/'
    files = os.listdir(filesPath)
    inputFiles = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.xlsx':# "文件获取"
            inputFiles.append(filesPath + files[i])

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('20点_20点')
    worksheet2 = workbook.add_sheet('8点_8点')


    c=0 #列变量
    for i in range(0, len(inputFiles)):#len(inputFiles)
        excel = xlrd.open_workbook(inputFiles[i])
        table = excel.sheet_by_index(0)
        rows = table.nrows
        r=1 #行变量,取1是为了留出列标题月份
        for j in range(1,rows):#注意跳过第一行的标题行

            if(j==1 or table.cell(j, 1).value != table.cell(j-1, 1).value): #控制跳过重复的数据
                temp=table.cell(j, 4).value
                worksheet.write(r, c, temp)

                temp = table.cell(j, 5).value
                worksheet2.write(r, c, temp)

                r=r+1
            else:
                continue

        c=c+1

    outFile = path+"outFile.xls"
    workbook.save(outFile)
#f1()

#   按照2016年7月份的89个站的站号，取出89个站的站点降水量。
def f2():
    path = 'F:/Test/Data/RGS/RGS83_201101_201712/'
    filesPath=path+'month/'
    files = os.listdir(filesPath)
    inputFiles = []
    for i in range(0, len(files)):
        if os.path.splitext(files[i])[1] == '.xlsx':# "文件获取"
            inputFiles.append(filesPath + files[i])

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('83RGS_20点_20点')



    excelRGS = xlrd.open_workbook(path+"RGSID.xlsx")
    tableRGS = excelRGS.sheet_by_index(0)
    rowsRGS = tableRGS.nrows

    c=0 #列变量
    for i in range(0, len(inputFiles)):#len(inputFiles)
        excel = xlrd.open_workbook(inputFiles[i])
        table = excel.sheet_by_index(0)
        rows = table.nrows
        # r=1 #行变量,取1是为了留出列标题月份
        for j in range(1,rowsRGS):#
            # if(r>=rowsRGS):#全部找到之后跳出循环
            #     break
            temp = 999999
            for k in range(1,rows):#注意跳过第一行的标题行
                if(table.cell(k, 1).value == tableRGS.cell(j, 1).value): #自动跳过重复行，因为r+1之后，重复项正好还是不相等
                    temp=table.cell(k, 4).value

                    # r=r+1
            worksheet.write(j, c, temp)
        c=c+1

    outFile = path+"outfile2.xls"
    workbook.save(outFile)
#f2()



