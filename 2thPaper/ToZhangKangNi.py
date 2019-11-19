# coding:gbk
import arcpy
from arcpy import env
from arcpy.sa import *
import sys, string, os
import xlrd
import xlwt
import numpy as np
import time
import math
import random
import sympy
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import random

def coarseGraining(field, coarseShape):
    # "����ۺ�ʱ�Ĵ��ڴ�С"
    rowRatio = sympy.S(field.shape[0]) / coarseShape[0]# "�����÷�����ʽ��ӣ��߽粻������"
    colRatio = sympy.S(field.shape[1]) / coarseShape[1]# "�����÷�����ʽ��ӣ��߽粻������"
    # print rowRatio
    # "ѭ�����㵱ǰ�㼶ÿ��������ȡֵ"
    # "��Է��������Ĵ����������Ʋ�ֵ�����ǣ�����ռԭ��������ı��������ۼӡ�"
    # "������window��ʵ��һ�����������window��field�����ʵ��ֻ�ǰ�λ�����"
    # "����i,j�����к��У������Ҷ�h��v�����˵�����Ҳ���к��У�Ȼ������field[math.floor(bottom):math.ceil(top),math.floor(left):math.ceil(right)]�ǵ�һ������Ϊ�У�"
    # "��һ������Ϊ�У���������������˳��Ҳ�õ���"
    # "����ֻ����������һ��ת�ò�����ʵ����window������field��ѭ���϶�������ˣ����к��У�"
    coarseField = np.zeros((coarseShape), np.float)

    top = 0
    for i in range(0, coarseShape[0]):

        bottom = top
        top = bottom + colRatio
        window_v = np.zeros(int(math.ceil(top) - math.floor(bottom)), np.float)
        for k in range(int(math.floor(bottom)), int(math.ceil(top))):
            if (k == int(math.floor(bottom))):
                window_v[k - int(math.floor(bottom))] = math.floor(bottom) + 1 - bottom
            elif (k == int(math.ceil(top) - 1)):
                window_v[k - int(math.floor(bottom))] = top + 1 - math.ceil(top)
            else:
                window_v[k - int(math.floor(bottom))] = 1
        window_v.shape = len(window_v), 1
        # print(window_v)

        right = 0
        for j in range(0, coarseShape[1]):
            left = right
            right = left + rowRatio
            window_h = np.zeros(int(math.ceil(right) - math.floor(left)), np.float)  #
            for k in range(int(math.floor(left)), int(math.ceil(right))):
                if (k == math.floor(left)):
                    window_h[k - int(math.floor(left))] = math.floor(left) + 1 - left
                elif (k == math.ceil(right) - 1):
                    window_h[k - int(math.floor(left))] = right + 1 - math.ceil(right)
                else:
                    window_h[k - int(math.floor(left))] = 1
            window_h.shape = 1, len(window_h)
            # print(window_h)

            window = window_v * window_h
            # print window
            # window = np.transpose(window)
            # �����������ˣ���*������˼�Ƕ�Ӧ��ˣ����ھ�����˵���Ǿ�����ˡ�
            coarseField[i, j] = np.sum(
                window * field[int(math.floor(bottom)):int(math.ceil(top)), int(math.floor(left)):int(math.ceil(right))])
            # print(coarseField[i, j])
    return coarseField
def exportPlot(logLambd,logMoment,k,b,q,alpha,f_alpha,taoq,D_q,pathAnalysis,fileName,pic,beta,sigma,e,v):
    #   ����������������
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(fileName)
    worksheet.write(0, 0, "log(Moments)&Log(lambda)")
    for i in range(0, len(logLambd)):
        worksheet.write(1, i + 1, logLambd[i])
        for j in range(0, len(q)):
            worksheet.write(j + 2, i + 1, logMoment[i, j])

    worksheet.write(len(q) + 2, 0, "q&Alpha&f_Alpha")
    for i in range(0, len(q)):
        worksheet.write(len(q) + 3, i + 1, q[i])
        worksheet.write(len(q) + 4, i + 1, alpha[i])
        worksheet.write(len(q) + 5, i + 1, f_alpha[i])
    worksheet.write(len(q) + 6, 0, "q&Taoq&D_q")
    for i in range(0, len(q)):
        worksheet.write(len(q) + 7, i + 1, q[i])
        worksheet.write(len(q) + 8, i + 1, taoq[i])
        worksheet.write(len(q) + 9, i + 1, D_q[i])
    worksheet.write(len(q) + 10, 0, "beta&sigma&e&v")
    worksheet.write(len(q) + 11, 1, beta)
    worksheet.write(len(q) + 11, 2, sigma)
    worksheet.write(len(q) + 11, 3, e)
    worksheet.write(len(q) + 11, 4, v)

    outExcel=pathAnalysis+fileName+'.xls'
    workbook.save(outExcel)


    # "�������ط�������ͼ"
    if pic == 'Y':

        # "��ͼlog2ͳ�ƾ���log2���ͳ߶�"
        plt.figure(1)
        x = logLambd
        for j in range(0, len(q)):
            plt.scatter(logLambd, logMoment[:,j])
            plt.plot(logLambd, k[j] * logLambd + b[j], "-")
            plt.text(logLambd[0], logMoment[:, j][0], 'q=' + str(q[j])[0:4], rotation=0)
        #      plt.text(x[-3], y[-3], 'q=' + str(q[j])[0:4] + ',$R^2=$' + str(rsquared[j])[0:6],rotation=-5)  # ��q��r2��ֵ��ʾ��ͼ�ϣ��Լ���ʾ��λ��
        # plt.xlim(-1, 6)
        # plt.ylim(-60, 100)
        plt.xlabel(r'$Log_2[\lambda]$')
        plt.ylabel(r'$Log_2[M(\lambda,q)]$')
        plt.savefig(pathAnalysis + "Moment_Lambda_" + fileName + ".png", dpi=300)
        # plt.show(1)
        plt.close(1)

        # "��ͼq��Alpha"
        plt.figure(2)
        plt.plot(q, alpha, "-o", label=fileName, color='blue')
        plt.plot((list(q)[0], list(q)[-1]), (list(alpha)[0], list(alpha)[-1]), color='red')
        plt.xlabel(r'q')
        plt.ylabel(r'${\alpha(q)}$')
        plt.savefig(pathAnalysis + 'q_alpha(q)_' + fileName + ".png", dpi=300)
        # plt.show(2)
        plt.close(2)

        # "��ͼAlpha��f_Alpha"
        plt.figure(3)
        plt.plot(alpha, f_alpha, "-o", label=fileName, color='blue')
        plt.xlabel(r'${\alpha}$')
        plt.ylabel(r"$f(\alpha)$")
        plt.savefig(pathAnalysis + "alpha_f(alpha)_" + fileName + ".png", dpi=300)
        # plt.show(3)
        plt.close(3)

        # "��ͼq��taoq"
        plt.figure(4)
        plt.plot(q, taoq, "-o", label=fileName, color='blue')
        plt.plot((list(q)[0], list(q)[-1]), (list(taoq)[0], list(taoq)[-1]), color='red')
        plt.xlabel(r'q')
        plt.ylabel(r"$\tau(q)$")
        plt.savefig(pathAnalysis + "q_tau(q)" + fileName + ".png", dpi=300)
        # plt.show(4)
        plt.close(4)

        # "��ͼq��D(q)"
        plt.figure(5)
        plt.plot(q, D_q, "-o", label=fileName, color='blue')
        plt.plot((list(q)[0], list(q)[-1]), (list(D_q)[0], list(D_q)[-1]), color='red')
        plt.xlabel(r'$q$')
        plt.ylabel(r'$D(q)$')
        # plt.legend(['data', 'linear', 'cubic'], loc='best')
        plt.savefig(pathAnalysis + "q_D(q)" + fileName + ".png", dpi=300)
        # plt.show(5)
        plt.close(5)

#   MF����
def MFRC(option,pic,pathIMERG, path3Mean, pathOutput, pathGWR,fileName):

    timeOfprocess = time.clock()

    # option='MF'   #'MFn'   'MFn-GWR'    ����ѡ��һ�ַ���
    # pic='Y' #   'N' �����Ƿ���������ͼ

    dimension = 2
    branch = 2
    m = 50  #��������Ĵ���
    n = 4   #ÿ�μ����Ĳ���
    # "�����޸ĵĲ���"

    fileName = fileName
    # path = pathIMERG
    pathIMERG= pathIMERG
    path3Mean= path3Mean

    pathResultRaster=pathOutput+'Rasters/'
    pathAnalysis=pathOutput+'Analysis/'
    #   �˴�Ӧ��GWR���ݵĲ���
    if (option=='MFn-GWR'):
        pathGWR=''

    q = np.linspace(-10, 10, 21)  # "qȡֵ��Χ"


    # ����դ������
    rasIMERG = arcpy.Raster(pathIMERG)
    ras3Mean = arcpy.Raster(path3Mean)

    # Set environmental variables for output
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = rasIMERG
    outResolution = 1000
    cellWidth = rasIMERG.meanCellWidth/16.0
    cellHeight = rasIMERG.meanCellHeight/16.0
    lowerLeft = arcpy.Point(rasIMERG.extent.XMin, rasIMERG.extent.YMin)

    # Convert Raster to numpy array
    arrIMERG = arcpy.RasterToNumPyArray(rasIMERG)
    arrIMERG = np.maximum(arrIMERG, 0)
    arr3Mean = arcpy.RasterToNumPyArray(ras3Mean)
    arr3Mean = np.maximum(arr3Mean, 0)

    if (option=='MFn-GWR'):
        rasGWR = arcpy.Raster(pathGWR)
        arrGWR = arcpy.RasterToNumPyArray(rasGWR)
        arrGWR = np.maximum(arrGWR, 0)

    # print arrIMERG,'\n\n\n',arr3Mean  #arrGWR

    # "���ʻ�"
    row = arrIMERG.shape[0]
    col = arrIMERG.shape[1]

    for i in range(0, row):
        for j in range(0, col):
            arr3Mean[i,j]=arr3Mean[i,j]/np.sum(arr3Mean)*(row*col)

# print row,col
    field = np.empty((row, col), np.float)
    for i in range(0, row):
        for j in range(0, col):
            if (arr3Mean[i, j] > 0):
                field[i, j] = arrIMERG[i, j] / arr3Mean[i, j]
            else:
                field[i, j] = 0
    print("field:", np.mean(arrIMERG))
    print("field:", np.mean(field))

    # field=arrIMERG

    # "��һ��"
    # sumField = np.sum(field)
    # if (sumField > 0):
    #     field = field / sumField
    # print field

    fieldSize = field.shape[0]
    # "layers+1 �����Ϸ����Ĳ�����scales��ÿ������Ԫ��С��Ӧ����ʼ0.1��ʱ�ı���"
    layers = np.arange(0, int(math.log(fieldSize, branch)))
    scales = branch ** layers
    # print("layers:", layers, "scales:", scales)

    # "��ͳ�ƾ�moment"
    # "d1��������taoq��һ�׵�(������ֵalpha�Ͳ���beta")��d2��������taoq�Ķ��׵��������sigma��"
    # "d3��������D(q)"
    moment = np.zeros((len(layers), len(q)))
    d1 = np.zeros((len(layers), len(q)))
    d2 = np.zeros((len(layers), len(q)))
    d3 = np.zeros((len(layers), len(q)))
    for i in range(0, len(layers)):
        distrib = coarseGraining(field, field.shape // scales[i])  ##[x // scales[i] for x infield.shape]
        positiveDist = distrib[distrib > 0]
        for j in range(0, len(q)):
            qmass = positiveDist ** q[j]
            moment[i, j] = np.sum(qmass)
            # print"distrib",distrib
            # print "q[j]",q[j]
            # print "moment[i,j]",moment[i,j]
            d1[i, j] = np.sum(qmass * np.log(positiveDist)) / np.sum(qmass)
            d2[i, j] = np.sum(qmass * np.log(positiveDist) ** 2) / np.sum(qmass) - d1[i, j] ** 2
            if (q[j] != 1):
                d3[i, j] = np.log(np.sum(qmass)) / (q[j] - 1)
            else:
                d3[i, j] = d1[i, j]

    lambd = 1.0/ branch ** (4-layers)
    print lambd
    logMoment = np.log(moment) / np.log(2)
    logLambd = np.log(lambd) / np.log(2)
    # "֤��������������"# "��tao(q),tao(q)����б��"
    k = np.zeros(len(q))  # ���б��
    b = np.zeros(len(q))  # ��Žؾ�
    # R_Squared = np.zeros(len(q))  # ���R��

    # x = np.log(lambd) / np.log(2)  # log��2Ϊ�׵�lambda,������С���˵�X����
    # X = sm.add_constant(X.T)  # ���Ͻؾ���
    for i in range(0, len(q)):
        # y = np.log(moment[:, i]) / np.log(2)  # log��2Ϊ�׵�moment��������С���˵�X����
        # results = sm.OLS(Y, X).fit()  # log��2Ϊ�׵�moment��lambda���������
        line = np.polyfit(logLambd, logMoment[:, i], 1)
        k[i] = line[0]  # б��
        b[i] = line[1]  # �ؾ�
        # R_Squared[i] = results.rsquared
        # print("k:", k[i], "b:", b[i], "Rsquared:", R_Squared[i])

    # "�ڶ��ط�������taoq���������б�ʣ��뼶�����߶��е�taoq��ͬ"
    taoq = -k
    # "֤�����ж��ط�������"#"�ö��ط�����f(��)��ʾ"
    alpha = np.zeros(len(q))
    f_alpha = np.zeros(len(q))
    for j in range(0, len(q)):
        line = np.polyfit(np.log(lambd), d1[:, j], 1)
        alpha[j] = line[0]
        f_alpha[j] = alpha[j] * q[j] -k[j]

    # "֤�����ж��ط�������"# "�ù������ά��D(q)��ʾ"
    D_q = np.zeros((len(q)))
    for j in range(0, len(q)):
        line = np.polyfit(np.log(lambd), d3[:, j], 1)
        D_q[j] = line[0]

    # "����׵����̶�����ºͦң���q=1����Ϊ����ֵ"
    # "���ڼ������߶ȵ��о��ߣ����γ߶���Ȼ��1��5��ǿ�аѸ����Ƶ���taoq�ϣ�������Xu�ȵĽ��߶��У�taoq�Ѿ��仯���������Ԧ˵ĸ���"
    # scales = scales[::-1]
    # print (scales)

    d1taoq  = -alpha
    d2taoq = np.zeros(len(q))
    for j in range(0, len(q)):
        d2[:, j] = d2[::, j]        #-��
        line = np.polyfit(np.log(lambd), d2[:, j], 1)
        d2taoq[j] = -line[0]
    # print 'taoq��һ�׵���',d1taoq,'\n','taoq�Ķ��׵���',d2taoq

    e = 0
    v = 1
    for i in range(0, len(q)):
        # print q[i]
        if (q[i] >= 1):
            if (option == "MF"):
                # "X�Ǳ�׼��̬�ֲ�"
                sigma = math.sqrt(d2taoq[i] / (dimension * np.log(branch**2)))
                beta = 1 + d1taoq[i] / dimension - sigma ** 2 * np.log(branch ** 2) * (q[i] - 0.5)
            else:
                # "X�ǷǱ�׼��̬�ֲ�"
                # "��Ҫ����ԭ���ݵľ�ֵ�ͷ���"
                data = np.array(arrIMERG).reshape(row * col, 1)
                d = []
                for j in range(0, row * col):
                    if data[j, 0] != 0:
                        d.append(data[j, 0])
                e = np.sum(np.log(d) / np.log(2)) / len(d)
                v = np.sum((np.log(d) / np.log(2) - e) ** 2) / len(d)
                # print("e:", e, "v:", v)
                sigma = math.sqrt(d2taoq[i] / (v*dimension * np.log(branch ** 2)))
                beta = 1 + d1taoq[i] / dimension - sigma ** 2 * np.log(branch ** 2) * (q[i] - 0.5)
            break
    print (beta, sigma, e, v)
    exportPlot(logLambd,logMoment,k,b,q,alpha,f_alpha,taoq,D_q,pathAnalysis,fileName,pic,beta,sigma,e,v)

    timeOfprocess=time.clock() - timeOfprocess
    print "���ط����������������������ʱ=", timeOfprocess, "��"

    # "ʵ�ʽ��߶Ȳ��õ����"
    fieldAll = []
    cascade = []
    for i in range(0, n + 1):
        cascade.append(np.zeros((branch ** i, branch ** i), np.double))
    # print ("cascade:",cascade)

    gamma = beta - sigma * e - v * sigma ** 2 * np.log(branch**2) / 2

    # "ѭ��m��"
    for k in range(0, m):
        for i in range(row):
            for j in range(col):
                cascade[0][0][0] = field[i, j]
                for x in range(1, n + 1):
                    for y in range(0, branch ** (x - 1)):
                        for z in range(0, branch ** (x - 1)):
                            w=np.zeros((4,1),float)
                            if (random.uniform(0,1)<=(branch**2)**(-beta)):
                                w[0]=(branch**2) ** (gamma + sigma * random.gauss(e, v))
                            else:
                                w[0]=0
                            if (random.uniform(0,1)<=(branch**2)**(-beta)):
                                w[1]=(branch**2) ** (gamma + sigma * random.gauss(e, v))
                            else:
                                w[1]=0
                            if (random.uniform(0,1)<=(branch**2)**(-beta)):
                                w[2]=(branch**2) ** (gamma + sigma * random.gauss(e, v))
                            else:
                                w[2]=0
                            w[3]=4-w[0]-w[1]-w[2]
                            # print w

                            cascade[x][y * 2][z * 2] = cascade[x - 1][y][z] * w[0]
                            cascade[x][y * 2][z * 2 + 1] = cascade[x - 1][y][z] * w[1]
                            cascade[x][y * 2 + 1][z * 2] = cascade[x - 1][y][z] * w[2]
                            cascade[x][y * 2 + 1][z * 2 + 1] = cascade[x - 1][y][z] * w[3]
                # simfield[:,.  :] = coarseGraining(cascade[n], (32, 32))
                # print("simfield:",simfield)
                if (j == 0):
                    fieldRow = cascade[n].copy()
                else:
                    fieldRow = np.hstack((fieldRow, cascade[n].copy()))
            if (i == 0):
                fieldMatrix = fieldRow.copy()
            else:
                fieldMatrix = np.vstack((fieldMatrix, fieldRow.copy()))
        # np.savetxt('F:/Test/OUT/'+"fieldAll"+str(k)+""+".txt",fieldMatrix,fmt = '%.8f')
        fieldAll.append(fieldMatrix)

    # "���ν��ƽ��ֵ"
    fieldAve = np.zeros((row * 2 ** n, col * 2 ** n), np.double)
    for k in range(0, m):
        fieldAve = fieldAve + fieldAll[k]
    fieldAve = fieldAve / m
    # np.savetxt('F:/Test/OUT/'+"fieldAve"+"ave"+".txt", fieldAve,fmt = '%.8f')

    # "�ָ�������"
    fieldHeter = np.zeros((row * 2 ** n, col * 2 ** n), np.double)
    for i in range(0, row):
        for j in range(0, col):
            if option == 'MFn-GWR':
                temp = arrGWR[i * 2 ** n:(i + 1) * 2 ** n, j * 2 ** n:(j + 1) * 2 ** n] \
                       * fieldAve[ i * 2 ** n:(i + 1) * 2 ** n, j * 2 ** n:(j + 1) * 2 ** n]
            else:
                temp = arr3Mean[i, j] * fieldAve[i * 2 ** n:(i + 1) * 2 ** n, j * 2 ** n:(j + 1) * 2 ** n]
            if (np.sum(temp) != 0):
                temp = temp / np.sum(temp)
            else:
                temp = 0
            fieldHeter[i * 2 ** n:(i + 1) * 2 ** n, j * 2 ** n:(j + 1) * 2 ** n] = temp * arrIMERG[i, j] * (
                    2 ** n * 2 ** n)
    # result = np.array(result).reshape(row * 2 ** n * col * 2 ** n, 1)
    # np.savetxt(path + 'out/' + "r" + option + ".txt", result, fmt='%.8f')

    # ������ݣ�����դ��
    tempRaster = arcpy.NumPyArrayToRaster(fieldHeter,lowerLeft,cellWidth,cellHeight)
    onekmRaster=pathResultRaster + 'r'+fileName+".tif"
    arcpy.Resample_management(tempRaster, onekmRaster, outResolution, "BILINEAR")   #"�ز�����1km"

    timeOfprocess=time.clock() - timeOfprocess
    print "���߶ȼ����ʱ=", timeOfprocess, "��"
def main():

    tt=time.clock()

    option = 'MF'  # 'MFn'   'MFn-GWR'    ����ѡ��һ�ַ���
    pic = 'Y'  # 'N' �����Ƿ���������ͼ

    for i in range(11,12):

        fileName=str((2015+i//12)*100+i%12+1)
        pathIMERG='F:/Test/Paper180829/Data/IMERG/10km8048/'+'I'+fileName+'.tif' #UTM8048    UTM9696
        # path3Mean='F:/Test/Paper180829/Data/IMERG/3Mean_UTMHB8048/'+str(i%12+1)+'.tif'    ##3Mean_UTM9696
        path3Mean = 'F:/Test/Paper180829/Data/IMERG/3Mean_UTMHB8048/' + str(i % 12 + 1) + '.tif'  ##3Mean_UTM9696
        pathOutput='F:/Test/Paper180829/Process/'+'0/'
        if (option == 'MF-GWR'):
            pathGWR=''
        else:
            pathGWR = ''
        MFRC(option,pic,pathIMERG, path3Mean, pathOutput, pathGWR,fileName)

    # points = 'F:/Test/Paper180829/data/POINT/' + 'UTMPOINTS83.shp'
    # pathRasters=pathOutput+'Rasters/'
    # arcpy.env.workspace =pathRasters
    # rasters=arcpy.ListRasters()
    #
    # for raster in rasters:
    #     out3=pathOutput + 'RG/' +raster[0:7]+'.shp'
    #     ExtractValuesToPoints(points, raster, out3, "INTERPOLATE", "VALUE_ONLY")  #"ֵ��ȡ����"
    #
    #     out4=pathOutput + 'Excels/' +raster[0:7]+'.xls'
    #     arcpy.TableToExcel_conversion(out3, out4)  # "��תExcel"
    #
    # print "�ܻ���ʱ��Ϊ��",time.clock()-tt,"��"\

main()