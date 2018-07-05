import numpy as np
import glob
import pandas as pd
import csv
from dbfread import DBF
#####TRMM每天的降雨值#####
#TRMM_DBF_Table = {}
#for Daily in range(1,32):#月数要改
#    TRMM_DBF_Table[Daily] = DBF('C:/Users/xia/Desktop/2006_2010/hbpoint/'+str(201012 * 100 + Daily)+"__hb"+'.dbf',load=True)#月数要改

#TRMM_TXT_Table = {}
#for Daily in range(1,32):#月数要改
#    TRMM_TXT_Table[Daily] = [0 for x in range(0, 353)]
#    for i in range(0,353):
#        TRMM_TXT_Table[Daily][i] = TRMM_DBF_Table[Daily].records[i]['GRID_CODE']
#    np.savetxt('C:/Users/xia/Desktop/2006_2010/hbpoint/'+ str(Daily + 1795)+'.txt',TRMM_TXT_Table[Daily],fmt = '%.8f')#月数要改


filelocation="C:\\Users\\xia\\Desktop\\2006_2010\\hbpoint_txt\\"
#当前文件夹下搜索的文件名后缀
fileform="txt"
#将合并后的表格存放到的位置
#filedestination="C:\\Users\\xia\\Desktop\\2006_2010\\hbpoint\\"
#合并后的表格命名为file
file_out=open('C:\\Users\\xia\\Desktop\\2006_2010\\20060101_20101231.txt','w')

filearray = []
for count in range(1, 1827):
    xx = glob.glob(filelocation+str(count)+"."+fileform)
    for filename in xx:
        filearray.append(filename)
#以上是从pythonscripts文件夹下读取所有txt，并将所有的名字存储到列表filearray
length_num = len(filearray)
print("在默认文件夹下有%d个文档哦"%length_num)

#HBTRMM_TXT_Table = []
#HBTRMM_TXT_Table = [0 for x in range(0, 1826)]
for i in range(length_num):
    fname = filearray[i]
    read_txt = open(fname)
    for line in read_txt:
        file_out.write(line)
    read_txt.close()
file_out.close()



