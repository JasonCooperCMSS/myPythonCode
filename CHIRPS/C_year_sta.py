# coding:utf-8
import numpy as np
import math
import csv
import arcpy
from arcpy import env
from dbfread import DBF
#####TRMM每天的降雨值#####
length=6
gauges=75
#        print (length,gauges)
TRMM_DBF_Table = {}
for year in range(1, length):
    TRMM_DBF_Table[year] = DBF(
        "F:\\CHIRPS\\CHIRPS\\year_hbRG\\" + str(2005+year) + "_RG.dbf",
        load=True)
TRMM_3B42_RG_Table = {}
for year in range(1, length):
    TRMM_3B42_RG_Table[year] = [0 for x in range(0, gauges)]  # 站点的数量200812与201001不一样

for year in range(1, length):
    for i in range(0, gauges):
        TRMM_3B42_RG_Table[year][i] = TRMM_DBF_Table[year].records[i]['RASTERVALU']
#        print(TRMM_3B42_RG_Table,'\n')

#####站点每天的降雨值#####
RG_DBF_Table = DBF("F:\\Research\\RainGauge\\ym_RG.dbf", load=True)
RG_Table = {}
for year in range(1, length):
    RG_Table[year] = [0 for x in range(0, gauges)]  # 站点的数量200812与201007不一样
for year in range(1, length):
    for j in range(0, gauges):
        RG_Table[year][j] = RG_DBF_Table.records[j]["Y"+str(2005+year)]
#        print(RG_Table)
#####R2，RMSE，MAE，Bias#####
TRMM_Data_Average = {}
RG_Data_Average = {}
sum1 = {}
sum2 = {}
sum3 = {}
sum4 = {}
sum5 = {}
sum6 = {}
sum7 = {}
R = {}
RMSE = {}
MAE = {}
Bias = {}
for year in range(1, length):
    TRMM_Data_Average[year] = sum(TRMM_3B42_RG_Table[year]) / len(TRMM_3B42_RG_Table[year])
    RG_Data_Average[year] = sum(RG_Table[year]) / len(RG_Table[year])
    sum1[year] = 0
    sum2[year] = 0
    sum3[year] = 0
    sum4[year] = 0
    sum5[year] = 0
    sum6[year] = 0
    sum7[year] = 0
    ###站点数量不同要适量调整###
    for n in range(0, gauges):
        sum1[year] = sum1[year] + (TRMM_3B42_RG_Table[year][n] - TRMM_Data_Average[year]) * (
        RG_Table[year][n] - RG_Data_Average[year])
        sum2[year] = sum2[year] + (TRMM_3B42_RG_Table[year][n] - TRMM_Data_Average[year]) ** 2
        sum3[year] = sum3[year] + (RG_Table[year][n] - RG_Data_Average[year]) ** 2
        sum4[year] = sum4[year] + (RG_Table[year][n] - TRMM_3B42_RG_Table[year][n]) ** 2
        sum5[year] = sum5[year] + abs(RG_Table[year][n] - TRMM_3B42_RG_Table[year][n])
        sum6[year] = sum6[year] + RG_Table[year][n]
        sum7[year] = sum7[year] + TRMM_3B42_RG_Table[year][n]
    if (sum2[year] == 0 or sum3[year] == 0 or sum4[year] == 0 or sum7[year] == 0):
#              print("分母有0值")
        R[year] = 999999
        RMSE[year] = 999999  ###站点数量不同要适量调整###
        Bias[year] = 999999
        MAE[year] = 999999  ###站点数量不同要适量调整###
    else:
        R[year] = sum1[year] / math.sqrt(sum2[year] * sum3[year])
        RMSE[year] = math.sqrt(sum4[year] / gauges)  ###站点数量不同要适量调整###
        Bias[year] = sum6[year] / sum7[year] - 1
        MAE[year] = sum5[year] / gauges  ###站点数量不同要适量调整###
        # print (str(R[year]))
sta = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\year\\sta_y.csv", "w"))
sta.writerow(['year','R','RMSE','Bias','MAE'])
for year in range(1,length):
    sta.writerow([str(2005+year),R[year],RMSE[year],Bias[year],MAE[year]])

####空演####
Count_KY = {}
Ratio_KY = {}
# 站点的数量200812与201007不一样
for i in range(0, gauges):
    Count_KY[i] = 0
    Ratio_KY[i] = 0.0
    for year in range(1, length):
        if (TRMM_3B42_RG_Table[year][i] != 0 and RG_Table[year][i] == 0):
            Count_KY[i] = Count_KY[i] + 1
            # else:
            # print('null')
    Ratio_KY[i] = float(Count_KY[i]) / (length-1)
# print(Count_KY,Ratio_KY)

####漏演####
Count_LY = {}
Ratio_LY = {}
for i in range(0, gauges):
    Count_LY[i] = 0
    Ratio_LY[i] = 0.0
    # 站点的数量200812与201007不一样
    for year in range(1, length):
        if (TRMM_3B42_RG_Table[year][i] == 0 and RG_Table[year][i] != 0):
            Count_LY[i] = Count_LY[i] + 1
            # else:
            # print('null')
    Ratio_LY[i] = float(Count_LY[i]) / (length-1)
#print(Count_LY, Ratio_LY)

#保存数据
staKL = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\year\\KL_y.csv", "w"))
staKL.writerow(['GaugeNum','Count_KY','Ratio_KY','Count_LY','Ratio_LY'])
for i in range(0,gauges):
    staKL.writerow([i,Count_KY[i],Ratio_KY[i],Count_LY[i], Ratio_LY[i]])

####适用度####
Ratio_SYD = {}
for year in range(1, length):
    Ratio_SYD[year] = [0 for x in range(0, gauges)]  # 站点的数量200812与201007不一样
for year in range(1, length):
    for i in range(0, gauges):
        if (RG_Table[year][i] != 0 and TRMM_3B42_RG_Table[year][i] != 0):
            Ratio_SYD[year][i] = np.abs(RG_Table[year][i] - TRMM_3B42_RG_Table[year][i]) / RG_Table[year][i]
            # print(Ratio_SYD)
        else:
            Ratio_SYD[year][i] = 999999
# print(Ratio_SYD)
#保存数据
staS = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\year\\SYD_y.csv", "w"))
staS.writerow(['SYD'])
for i in range(1, length):
    staS.writerow(Ratio_SYD[i])