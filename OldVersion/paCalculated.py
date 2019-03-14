import multiprocessing
import numpy as np
from dbfread import DBF
import time
Downscaling_DBF_Table = DBF('C:/Users/xia/Desktop/2006_2010/200901/20090105/20090105_gauss01_mf_50Ave.dbf', load=True)
lenDownscaling = len(Downscaling_DBF_Table)

def f(x,y,z):
    index = 0
    mindistance2 = 10000000
    for j in range(0,lenDownscaling):
        distance2 = (x - float(Downscaling_DBF_Table.records[j]['x'])) ** 2 + (y - float(Downscaling_DBF_Table.records[j]['y'])) ** 2
        if (mindistance2 > distance2):
            mindistance2 = distance2
            index = j
        elif (mindistance2 == distance2):
            if (np.abs(Downscaling_DBF_Table.records[index]['Field1'] - z) >= np.abs(Downscaling_DBF_Table.records[j]['Field1'] - z)):
                index = j
    #return index
    return Downscaling_DBF_Table.records[index]['Field1']

#def f2(a):
#    return GLOBAL_A


# def calculated( ):



if __name__ == '__main__':


    ########输入数据########
    #RG_DBF_Table = DBF('E:/研究生/Data/气象降水数据/RAIN/200901.dbf',load=True)
    CmorphRH_DBF_Table = DBF('C:/Users/xia/Desktop/2006_2010/200901/20090105/20090105_cmorphrh_hb.dbf',load=True)
    #TRMM_DBF_Table = DBF('C:/Users/xia/Desktop/2006_2010/200901/20090105_trmm_3232.dbf',load=True)
    #Downscaling_DBF_Table = DBF('C:/Users/xia/Desktop/2006_2010/200901/20090105/20090105_gaussUS_Gmf_50Ave.dbf',load=True)
    #lenRG = len(RG_DBF_Table)
    lenCmorphRH = len(CmorphRH_DBF_Table)
    #lenTRMM = len(TRMM_DBF_Table)
    #lenDownscaling = len(Downscaling_DBF_Table)
    #RasterValue = [0 for x in range(0, lenRG)]
    #RasterValue = [0 for x in range(0, lenCmorphRH)]
    greenPointX = [0 for x in range(0, lenCmorphRH)]
    greenPointY = [0 for x in range(0, lenCmorphRH)]
    greenPointZ = [0 for x in range(0, lenCmorphRH)]
    greenPoint = np.empty((lenCmorphRH,3))
    bluePoint = [0 for x in range(0, lenCmorphRH)]
    for i in range(0,lenCmorphRH): #lenCmorphRH
        greenPointX[i] = CmorphRH_DBF_Table.records[i]['x']
        greenPointY[i] = CmorphRH_DBF_Table.records[i]['y']
        greenPointZ[i] = CmorphRH_DBF_Table.records[i]['GRID_CODE']
        greenPoint[i] = [greenPointX[i],greenPointY[i],greenPointZ[i]]

    cores = multiprocessing.cpu_count()
    # start a pool
    pool = multiprocessing.Pool(processes=cores)
    bluePoint = pool.starmap(f, greenPoint)
    np.savetxt('C:/Users/xia/Desktop/2006_2010/200901/20090105/'+str(20090105)+'_gauss01_mf_RG'+'.txt',bluePoint,fmt='%.8f')

