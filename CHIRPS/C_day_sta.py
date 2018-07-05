# coding:utf-8
import numpy as np
import math
import csv
import arcpy
from arcpy import env
from dbfread import DBF
for yy in range(10,11):
    for mm in range(7,8):
        #####TRMM每天的降雨值#####
        arcpy.env.workspace = "F:\\CHIRPS\\CHIRPS\\day_hbRG\\201007\\"
        days= arcpy.ListFeatureClasses()
        length=len(days)+1
        gauges=75
        if (yy==8 and mm==12):
            gauges=63
        if (yy==10 and mm==1):
            gauges=71
#        print (length,gauges)

        TRMM_DBF_Table = {}
        for Daily in range(1, length):
            TRMM_DBF_Table[Daily] = DBF(
                "F:\\CHIRPS\\CHIRPS\\day_hbRG\\"+str((2000+yy)*100+mm)+"\\" + str((2000+yy)*10000+mm*100+Daily) + "_RG" + ".dbf",
                load=True)

        TRMM_3B42_RG_Table = {}
        for Daily in range(1, length):
            TRMM_3B42_RG_Table[Daily] = [0 for x in range(0, gauges)]  # 站点的数量200812与201001不一样

        for Daily in range(1, length):
            for i in range(0, gauges):
                TRMM_3B42_RG_Table[Daily][i] = TRMM_DBF_Table[Daily].records[i]['RASTERVALU']
#        print(TRMM_3B42_RG_Table,'\n')

        #####站点每天的降雨值#####
        RG_DBF_Table = DBF("F:\\Research\\RainGauge\\RG_dbf\\" + str((2000+yy)*100+mm) + ".dbf", load=True)
        RG_Table = {}
        for Daily in range(1, length):
            RG_Table[Daily] = [0 for x in range(0, gauges)]  # 站点的数量200812与201007不一样
        for Daily in range(1, length):
            for j in range(0, gauges):
                RG_Table[Daily][j] = RG_DBF_Table.records[j]["a" + str(Daily)]
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
        for Daily in range(1, length):
            TRMM_Data_Average[Daily] = sum(TRMM_3B42_RG_Table[Daily]) / len(TRMM_3B42_RG_Table[Daily])
            RG_Data_Average[Daily] = sum(RG_Table[Daily]) / len(RG_Table[Daily])
            sum1[Daily] = 0
            sum2[Daily] = 0
            sum3[Daily] = 0
            sum4[Daily] = 0
            sum5[Daily] = 0
            sum6[Daily] = 0
            sum7[Daily] = 0
            ###站点数量不同要适量调整###
            for n in range(0, gauges):
                sum1[Daily] = sum1[Daily] + (TRMM_3B42_RG_Table[Daily][n] - TRMM_Data_Average[Daily]) * (
                RG_Table[Daily][n] - RG_Data_Average[Daily])
                sum2[Daily] = sum2[Daily] + (TRMM_3B42_RG_Table[Daily][n] - TRMM_Data_Average[Daily]) ** 2
                sum3[Daily] = sum3[Daily] + (RG_Table[Daily][n] - RG_Data_Average[Daily]) ** 2
                sum4[Daily] = sum4[Daily] + (RG_Table[Daily][n] - TRMM_3B42_RG_Table[Daily][n]) ** 2
                sum5[Daily] = sum5[Daily] + abs(RG_Table[Daily][n] - TRMM_3B42_RG_Table[Daily][n])
                sum6[Daily] = sum6[Daily] + RG_Table[Daily][n]
                sum7[Daily] = sum7[Daily] + TRMM_3B42_RG_Table[Daily][n]
            if (sum2[Daily] == 0 or sum3[Daily] == 0 or sum4[Daily] == 0 or sum7[Daily] == 0):
  #              print("分母有0值")
                R[Daily] = 999999
                RMSE[Daily] = 999999  ###站点数量不同要适量调整###
                Bias[Daily] = 999999
                MAE[Daily] = 999999  ###站点数量不同要适量调整###
            else:
                R[Daily] = sum1[Daily] / math.sqrt(sum2[Daily] * sum3[Daily])
                RMSE[Daily] = math.sqrt(sum4[Daily] / gauges)  ###站点数量不同要适量调整###
                Bias[Daily] = sum6[Daily] / sum7[Daily] - 1
                MAE[Daily] = sum5[Daily] / gauges  ###站点数量不同要适量调整###
                # print (str(R[Daily]))
        sta = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\day\\"+str((2000+yy)*100+mm)+"_sta_daily.csv", "w"))
        sta.writerow(['Day','R','RMSE','Bias','MAE'])
        for Daily in range(1,length):
            sta.writerow([Daily,R[Daily],RMSE[Daily],Bias[Daily],MAE[Daily]])

        ####空演####
        Count_KY = {}
        Ratio_KY = {}
        # 站点的数量200812与201007不一样
        for i in range(0, gauges):
            Count_KY[i] = 0
            Ratio_KY[i] = 0.0
            for daily in range(1, length):
                if (TRMM_3B42_RG_Table[daily][i] != 0 and RG_Table[daily][i] == 0):
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
            for daily in range(1, length):
                if (TRMM_3B42_RG_Table[daily][i] == 0 and RG_Table[daily][i] != 0):
                    Count_LY[i] = Count_LY[i] + 1
                    # else:
                    # print('null')
            Ratio_LY[i] = float(Count_LY[i]) / (length-1)
        #print(Count_LY, Ratio_LY)

        #保存数据
        staKL = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\day\\"+str((2000+yy)*100+mm)+"_KL_daily.csv", "w"))
        staKL.writerow(['GaugeNum','Count_KY','Ratio_KY','Count_LY','Ratio_LY'])
        for i in range(0,gauges):
            staKL.writerow([i,Count_KY[i],Ratio_KY[i],Count_LY[i], Ratio_LY[i]])

        ####适用度####
        Ratio_SYD = {}
        for Daily in range(1, length):
            Ratio_SYD[Daily] = [0 for x in range(0, gauges)]  # 站点的数量200812与201007不一样
        for Daily in range(1, length):
            for i in range(0, gauges):
                if (RG_Table[Daily][i] != 0 and TRMM_3B42_RG_Table[Daily][i] != 0):
                    Ratio_SYD[Daily][i] = np.abs(RG_Table[Daily][i] - TRMM_3B42_RG_Table[Daily][i]) / RG_Table[Daily][i]
                    # print(Ratio_SYD)
                else:
                    Ratio_SYD[Daily][i] = 999999
        # print(Ratio_SYD)
        #保存数据
        staS = csv.writer(open("F:\\CHIRPS\\CHIRPS\\RainGauges\\statistics\\day\\"+str((2000+yy)*100+mm)+"_SYD_daily.csv", "w"))
        staS.writerow(['SYD'])
        for i in range(1, length):
            staS.writerow(Ratio_SYD[i])