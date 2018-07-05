# coding:utf-8
import xlrd
import xlwt
import sys, string, os
path = 'F:/SA/RAIN/RAIN/month/'
files = os.listdir(path)
excel_file=[]
for i in range(0,len(files)):
    if os.path.splitext(files[i])[1] == '.xlsx':
        excel_file.append(path+files[i])

data="F:/SA/RAIN/RAIN/"+"TEMP.xlsx"
excel= xlrd.open_workbook(data)
table=excel.sheet_by_index(0)
rows=table.nrows
cols=table.ncols
#print rows,cols

workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('11_17')

for x in range(0,84):
    worksheet.write(0, x, str(x))
    excel1 = xlrd.open_workbook(excel_file[x])
    table1 = excel1.sheet_by_index(0)
    rows1 = table1.nrows
    cols1 = table1.ncols
#    print rows1,cols1

    for i in range(0, rows):
        y = 0
        for j in range(1, rows1):
            #        print table.cell(j,1).ctype,table1.cell(i,0).ctype
            #       print table.cell(j, 1).value, table1.cell(i, 0).value
            if (int(table1.cell(j, 1).value) == int(table.cell(i, 0).value)):
                if (y==0):
                    y=y+1
                else:
                    break
                worksheet.write(i+1, x, float(table1.cell(j, 4).value))

out="F:/SA/RAIN/RAIN/"+"haha.xls"
workbook.save(out)


