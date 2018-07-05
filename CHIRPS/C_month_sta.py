# coding:utf-8
import numpy as np
import math
import csv
import arcpy
from arcpy import env
from dbfread import DBF
for yy in range(6,11):
    #####TRMM每天的降雨值#####
    arcpy.env.workspace = "F:\\CHIRPS\\CHIRPS\\month_hbRG\\"+str(2000+yy)+"\\"
    days= arcpy.ListFeatureClasses()
    length=13
    gauges = 75
#        print (length,gauges)

    TRMM_DBF_Table = {}
    for month in range(1, length):
        TRMM_DBF_Table[month] = DBF(
            "F:\\CHIRPS\\CHIRPS\\month_hbRG\\"+str(2000+yy)+"\\" + str((2000+yy)*100+month) + "_RG.dbf",
            load=True)
    TRMM_3B42_RG_Table = {}
    for month in range(1, length):
        TRMM_3B42_RG_Table[month] = [0 for x in range(0, gauges)]  # 站点的数量200812与201001不一样

    for month in range(1, length):
        for i in range(0, gauges):
            TRMM_3B42_RG_Table[month][i] = TRMM_DBF_Table[month].records[i]['RASTERVALU']
#        print(TRMM_3B42_RG_Table,'\n')

    #####站点每天的降雨值#####
    RG_DBF_Table = DBF("F:\\Research\\RainGauge\\ym_RG.dbf", load=True)
    RG_Table = {}
    for month in range(1, length):
        RG_Table[month] = [0 for x in range(0, gauges)]  # 站点的数量200812与201007不一样
    for month in range(1, length):
        for j in range(0, gauges):
            RG_Table[month][j] = RG_DBF_Table.records[j]["M"+str((2000+yy)*100+month)]
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
    for month in range(1, length):
        TRMM_Data_Average[month] = sum(TRMM_3B42_RG_Table[month]) / len(TRMM_3B42_RG_Table[month])
        RG_Data_Average[month] = sum(RG_Table[month]) / len(RG_Table[month])
        sum1[month] = 0
        sum2[month] = 0
        sum3[month] = 0
        sum4[month] = 0
        sum5[month] = 0
        sum6[month] = 0
        sum7[month] = 0
        ###站点数量不同要适量调整###
        for n in range(0, gauges):
            sum1[month] = sum1[month] + (TRMM_3B42_RG_Table[month][n] - TRMM_Data_Average[month]) * (
            RG_Table[month][n] - RG_Data_Average[month])
            sum2[month] = sum2[month] + (TRMM_3B42_RG_Table[month][n] - TRMM_Data_Average[month]) ** 2
            sum3[month] = sum3[month] + (RG_Table[month][n] - RG_Data_Average[month]) ** 2
            sum4[month] = sum4[month] + (RG_Table[month][n] - TRMM_3B42_RG_Table[month][n]) ** 2
            sum5[month] = sum5[month] + abs(RG_Table[month][n] - TRMM_3B42_RG_Table[month][n])
            sum6[month] = sum6[month] + RG_Table[month][n]
            sum7[month] = sum7[month] + TRMM_3B42_RG_Table[month][n]
        if (sum2[month] == 0 or sum3[month] == 0 or sum4[month] == 0 or sum7[month] == 0):
#              print("分母有0值")
            R[month] = 999999
            RMSE[month] = 999999  ###站点数量不同要适量调整###
            Bias[month] = 999999
            MAE[month] = 999999  ###站点数量不同要适量调整###
        else:
            R[month] = sum1[month] / math.sqrt(sum2[month] * sum3[month])
            RMSE[month] = math.sqrt(sum4[month] / gauges)  ###站点数量不同要适量调整###
            Bias[month] = sum6[month] / sum7[month] - 1
            MAE[month] = sum5[month] / gauges  ###站点数量不同要适量调整###
            # print (str(R[month]))
    sta = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\month\\"+str(2000+yy)+"_sta_m.csv", "w"))
    sta.writerow(['Month','R','RMSE','Bias','MAE'])
    for month in range(1,length):
        sta.writerow([month,R[month],RMSE[month],Bias[month],MAE[month]])

    ####空演####
    Count_KY = {}
    Ratio_KY = {}
    # 站点的数量200812与201007不一样
    for i in range(0, gauges):
        Count_KY[i] = 0
        Ratio_KY[i] = 0.0
        for month in range(1, length):
            if (TRMM_3B42_RG_Table[month][i] != 0 and RG_Table[month][i] == 0):
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
        for month in range(1, length):
            if (TRMM_3B42_RG_Table[month][i] == 0 and RG_Table[month][i] != 0):
                Count_LY[i] = Count_LY[i] + 1
                # else:
                # print('null')
        Ratio_LY[i] = float(Count_LY[i]) / (length-1)
    #print(Count_LY, Ratio_LY)

    #保存数据
    staKL = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\month\\"+str(2000+yy)+"_KL_m.csv", "w"))
    staKL.writerow(['GaugeNum','Count_KY','Ratio_KY','Count_LY','Ratio_LY'])
    for i in range(0,gauges):
        staKL.writerow([i,Count_KY[i],Ratio_KY[i],Count_LY[i], Ratio_LY[i]])

    ####适用度####
    Ratio_SYD = {}
    for month in range(1, length):
        Ratio_SYD[month] = [0 for x in range(0, gauges)]  # 站点的数量200812与201007不一样
    for month in range(1, length):
        for i in range(0, gauges):
            if (RG_Table[month][i] != 0 and TRMM_3B42_RG_Table[month][i] != 0):
                Ratio_SYD[month][i] = np.abs(RG_Table[month][i] - TRMM_3B42_RG_Table[month][i]) / RG_Table[month][i]
                # print(Ratio_SYD)
            else:
                Ratio_SYD[month][i] = 999999
    # print(Ratio_SYD)
    #保存数据
    staS = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\month\\"+str(2000+yy)+"_SYD_m.csv", "w"))
    staS.writerow(['SYD'])
    for i in range(1, length):
        staS.writerow(Ratio_SYD[i])