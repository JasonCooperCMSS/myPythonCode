# coding:utf-8
import numpy as np
import math
import csv
import arcpy
from arcpy import env
from dbfread import DBF
#####TRMM每天的降雨值#####
TRMM_DBF_Table = {}
for Daily in range(1, 32):
    TRMM_DBF_Table[Daily] = DBF( "F:\\Research\\2006_2010\\day_hbRG\\200601_RG\\"+ str(200601*100+Daily)+ "_RG" + ".dbf",load=True)

TRMM_3B42_RG_Table = {}
for Daily in range(1, 32):
    TRMM_3B42_RG_Table[Daily] = [0 for x in range(0, 75)]  # 站点的数量200812与201001不一样

for Daily in range(1, 32):
    for i in range(0, 75):
        TRMM_3B42_RG_Table[Daily][i] = TRMM_DBF_Table[Daily].records[i]['RASTERVALU']
# print(TRMM_3B42_RG_Table)
#####站点每天的降雨值#####
RG_DBF_Table = DBF("F:\\Research\\RainGauge\\RG_dbf\\" + str(200601) + ".dbf", load=True)
RG_Table = {}
for Daily in range(1, 32):
    RG_Table[Daily] = [0 for x in range(0, 75)]  # 站点的数量200812与201007不一样
for Daily in range(1, 32):
    for j in range(0, 75):
        RG_Table[Daily][j] = RG_DBF_Table.records[j]["a" + str(Daily)]
# print(RG_Table)
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
for Daily in range(1, 32):
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
    for n in range(0, 75):
        sum1[Daily] = sum1[Daily] + (TRMM_3B42_RG_Table[Daily][n] - TRMM_Data_Average[Daily]) * (
        RG_Table[Daily][n] - RG_Data_Average[Daily])
        sum2[Daily] = sum2[Daily] + (TRMM_3B42_RG_Table[Daily][n] - TRMM_Data_Average[Daily]) ** 2
        sum3[Daily] = sum3[Daily] + (RG_Table[Daily][n] - RG_Data_Average[Daily]) ** 2
        sum4[Daily] = sum4[Daily] + (RG_Table[Daily][n] - TRMM_3B42_RG_Table[Daily][n]) ** 2
        sum5[Daily] = sum5[Daily] + abs(RG_Table[Daily][n] - TRMM_3B42_RG_Table[Daily][n])
        sum6[Daily] = sum6[Daily] + RG_Table[Daily][n]
        sum7[Daily] = sum7[Daily] + TRMM_3B42_RG_Table[Daily][n]
    if (sum2[Daily] == 0 or sum3[Daily] == 0 or sum4[Daily] == 0 or sum7[Daily] == 0):
        print("分母有0值")
    else:
        R[Daily] = sum1[Daily] / math.sqrt(sum2[Daily] * sum3[Daily])
        RMSE[Daily] = math.sqrt(sum4[Daily] / 75)  ###站点数量不同要适量调整###
        Bias[Daily] = sum6[Daily] / sum7[Daily] - 1
        MAE[Daily] = sum5[Daily] / 75  ###站点数量不同要适量调整###
        # print (str(R[Daily]))
####保存数据####
arcpy.CreateFolder_management("F:\\Research\\RainGauge\\Statistics\\daily\\", "200601")
W_R = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_R.csv", "w"))
for key_R, val_R in R.items():
    W_R.writerow([key_R, val_R])
W_RMSE = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_RMSE.csv", "w"))
for key_RMSE, val_RMSE in RMSE.items():
    W_RMSE.writerow([key_RMSE, val_RMSE])
W_MAE = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_MAE.csv", "w"))
for key_MAE, val_MAE in MAE.items():
    W_MAE.writerow([key_MAE, val_MAE])
W_Bias = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_Bias.csv", "w"))
for key_Bias, val_Bias in Bias.items():
    W_Bias.writerow([key_Bias, val_Bias])
####空演####
Count_KY = {}
Ratio_KY = {}
# 站点的数量200812与201007不一样
for i in range(0, 75):
    Count_KY[i] = 0
    Ratio_KY[i] = 0
    for daily in range(1, 31):
        if (TRMM_3B42_RG_Table[daily][i] != 0 and RG_Table[daily][i] == 0):
            Count_KY[i] = Count_KY[i] + 1
            # else:
            # print('null')
    Ratio_KY[i] = Count_KY[i] / 31
# print(Count_KY,Ratio_KY)
# 保存空演数据#
W_Count_KY = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_Count_KY.csv", "w"))
for key_Count_KY, val_Count_KY in Count_KY.items():
    W_Count_KY.writerow([key_Count_KY, val_Count_KY])
W_Ratio_KY = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_Ratio_KY.csv", "w"))
for key_Ratio_KY, val_Ratio_KY in Ratio_KY.items():
    W_Ratio_KY.writerow([key_Ratio_KY, val_Ratio_KY])

####漏演####
Count_LY = {}
Ratio_LY = {}
for i in range(0, 75):
    Count_LY[i] = 0
    Ratio_LY[i] = 0
    # 站点的数量200812与201007不一样
    for daily in range(1, 32):
        if (TRMM_3B42_RG_Table[daily][i] == 0 and RG_Table[daily][i] != 0):
            Count_LY[i] = Count_LY[i] + 1
            # else:
            # print('null')
    Ratio_LY[i] = Count_LY[i] / 31
# 保存漏演数据#
W_Count_LY = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_Count_LY.csv", "w"))
for key_Count_LY, val_Count_LY in Count_LY.items():
    W_Count_LY.writerow([key_Count_LY, val_Count_LY])
W_Ratio_LY = csv.writer(open("F:\\Research\\RainGauge\\Statistics\\daily\\200601\\200601_Ratio_LY.csv", "w"))
for key_Ratio_LY, val_Ratio_LY in Ratio_LY.items():
    W_Ratio_LY.writerow([key_Ratio_LY, val_Ratio_LY])

####适用度####
# Ratio_SYD = {}
# 站点的数量200812与201007不一样
Ratio_SYD = np.empty((31,75))
# for i in range(0,75):
#    if(RG_Table[10][i] != 0 and TRMM_3B42_RG_Table[10][i] != 0):
#        Ratio_SYD[10][i] = np.abs(RG_Table[10][i] - TRMM_3B42_RG_Table[10][i])/ RG_Table[10][i]
#            #print(Ratio_SYD)
#    else:
#        Ratio_SYD[10][i] = 999999
# np.savetxt('E:/201007/适用度/' + str(20100710) +'_SYD'+ '.' + "txt", Ratio_SYD[10], fmt='%.8f')
for Daily in range(1, 32):
    for i in range(0, 75):
        if (RG_Table[Daily][i] != 0 and TRMM_3B42_RG_Table[Daily][i] != 0):
            Ratio_SYD[Daily][i] = np.abs(RG_Table[Daily][i] - TRMM_3B42_RG_Table[Daily][i]) / RG_Table[Daily][i]
            # print(Ratio_SYD)
        else:
            Ratio_SYD[Daily][i] = 999999
    np.savetxt('F:\\Research\\RainGauge\\Statistics\\daily\\200601\\' + str(
        200601 * 100 + Daily) + '_SYD' + '.' + "txt", Ratio_SYD[Daily], fmt='%.8f')
    # print(Ratio_SYD)
    # 保存适用度数据#
    # W_Ratio_SYD = csv.writer(open("E:/200812/200812_Ratio_SYD.csv","w"))
    # for key_Ratio_SYD, val_Ratio_SYD in Ratio_SYD.items():
    #    W_Ratio_SYD.writerow([key_Ratio_SYD, val_Ratio_SYD])
