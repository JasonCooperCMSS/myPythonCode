# coding 'utf-8'
# zsj 20180705
import numpy as np
import math
import sympy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import leastsq
import random
import copy

def coarseGrain(field, coarseShape):
    rowFine = field.shape[0]
    rowRatio = sympy.S(rowFine) / coarseShape[0]
    coarseField = np.zeros((coarseShape), np.float)

    "��ά"
    collumFine = field.shape[1]
    collumRatio = sympy.S(collumFine) / coarseShape[1]
    right = 0
    for i in range(0, coarseShape[0]):
        left = right
        right = left + rowRatio
        window_h = np.zeros((math.ceil(right) - math.floor(left)), np.float)

        for k in range(int(math.floor(left)), int(math.ceil(right))):
            if (k == math.floor(left)):
                window_h[k - math.floor(left)] = math.floor(left) + 1 - left
                # print(window_h[k-math.floor(left)])
            elif (k == math.ceil(right) - 1):
                window_h[k - math.floor(left)] = right + 1 - math.ceil(right)
                # print(window_h[k-math.floor(left)])
            else:
                window_h[k - math.floor(left)] = 1
        window_h.shape = 1, len(window_h)
        # print(window_h)
        top = 0
        for j in range(0, coarseShape[1]):
            bottom = top
            top = bottom + collumRatio

            window_v = np.zeros((math.ceil(top) - math.floor(bottom)), np.float)
            for k in range(int(math.floor(bottom)), int(math.ceil(top))):
                if (k == math.floor(bottom)):
                    window_v[k - math.floor(bottom)] = math.floor(bottom) + 1 - bottom
                elif (k == math.ceil(top) - 1):
                    window_v[k - math.floor(bottom)] = top + 1 - math.ceil(top)
                else:
                    window_v[k - math.floor(bottom)] = 1
            # print(window_v)
            window_v.shape = len(window_v), 1
            window = window_h * window_v
            window = np.transpose(window)
            coarseField[i, j] = np.sum(
                window * field[math.floor(left):math.ceil(right), math.floor(bottom):math.ceil(top)])

    # print(coarseField)
    return coarseField


def normalize(field):
    sumField = np.sum(field)
    if (sumField > 0):
        return field / sumField
    else:
        return field





"���õ±仯����ط���ά��f(a)"


def legendre(rawfield, branch=2):
    field = normalize(rawfield)
    fieldSize = field.shape[0]
    layer = np.arange(0, int(math.log(fieldSize, branch)))
    scale = branch ** layer
    q = np.linspace(-5, 5, 11)
    alpha = np.zeros((len(scale), len(q)))
    f_alpha = np.zeros((len(scale), len(q)))
    a = np.zeros((len(q)))
    f = np.zeros((len(q)))
    for i in range(0, len(scale)):
        for j in range(0, len(q)):
            distrib = coarseGrain(field, [x // scale[i] for x in field.shape])
            positiveDist = distrib[distrib > 0]
            qmass = positiveDist ** q[j]
            alpha[i, j] = np.sum((qmass * np.log(positiveDist)) / np.sum(qmass))
            # f_alpha[i,j]=(q[j]*np.sum(qmass*np.log(positiveDist))-np.sum(qmass)*np.log(np.sum(qmass)))/np.sum(qmass)
            f_alpha[i, j] = q[j] * alpha[i, j] - np.log(np.sum(qmass))
    for j in range(0, len(q)):
        line1 = np.polyfit(np.log(1.0 * scale), alpha[:, j], 1)
        line2 = np.polyfit(np.log(1.0 * scale), f_alpha[:, j], 1)
        a[j] = line1[0]
        f[j] = line2[0]

    plotSpectrum(a, f)

    print("a:", a, "f:", f)
    return (a, f)


"����log��2Ϊ�׵�moment��lambda��ͼ"

def plotMoment_lambd(scale, q, k, b, moment, rsquared, name=""):
    plt.figure(1)
    lambd = scale
    # lambd = scale[::-1]
    x = np.log(lambd) / np.log(2)
    for j in range(0, moment.shape[1]):
        y = np.log(moment[:, j]) / np.log(2)
        plt.plot(x, y,"-x")
        plt.plot(x, k[j] * x + b[j], "-")
        plt.text(x[-2], y[0], 'q=' + str(q[j])[0:4], rotation=-5)
        print("x:", x, "y:", y)
#        plt.text(x[-3], y[-3], 'q=' + str(q[j])[0:4] + ',$R^2=$' + str(rsquared[j])[0:6],rotation=-5)  # ��q��r2��ֵ��ʾ��ͼ�ϣ��Լ���ʾ��λ��
    plt.xlim(-2, 6)
    plt.ylim(-500, 500)
    plt.xlabel(r'$Log_2[\lambda]$')
    plt.ylabel(r'$Log_2[M(\lambda,q)]$')
    # plt.savefig('C:/Users/xia/Desktop/20060101vs20060622vs20061025/TRMM3b43200606/'+"momentscale20060625256"+name+".png",dpi=300)
    # plt.savefig('C:/Users/xia/Desktop/2006_2010/201007/' + "momentscale2010070832323" + name + ".png", dpi=300)###���
    # plt.close(1)


def plotD_q(k, q, name=""):
    plt.figure(2)
    taoq = -k
    d1_taoq = (taoq[1:] - taoq[0:-1]) / (q[1:] - q[0:-1])
    for i in range(0, len(q)):
        if (q[i] != 1):
            D_q = taoq / (q[i] - 1)
        else:
            D_q = d1_taoq[i]
    print("D(q):",D_q)
    # plt.axis([-5, -2,5 ,4])
    plt.plot(q, D_q, "-o", label=name,color='black')
    plt.plot((list(q)[0], list(q)[-1]), (list(D_q)[0], list(D_q)[-1]),color='red')
    plt.xlabel(r'$q$')
    plt.ylabel(r'$D(q)$')
#    plt.legend(['data', 'linear', 'cubic'], loc='best')
#     plt.savefig('C:/Users/xia/Desktop/2006_2010/201007/' + "Dq2010070832322" + name + ".png", dpi=300)
    # plt.close(2)


"�����ۻ�����ͼ"


def plotSpectrum(a, f, name=""):
    plt.figure(3)
    # plt.axis([0.25,0.3,3,2])
    plt.plot(a, f, '-o', label=name,color='black')
    plt.xlabel(r'${\alpha}$')
    plt.ylabel(r"$f(\alpha)$")
    #plt.title("���ط�����")
    plt.legend()
    # plt.savefig('C:/Users/xia/Desktop/20060101vs20060622vs20061025/TRMM3b43200606/'+"Spectrum20060625256"+name+".png",dpi=300)
    # plt.savefig('C:/Users/xia/Desktop/2006_2010/201007/' + "Spectrum2010070832322" + name + ".png", dpi=300)##���
    # plt.close(3)


"����taoq��q��beta��sigma"


#def lognormalBetaFit(q, taoq):
#    d = 2
#    b = 2
#    d1_taoq = (taoq[1:] - taoq[0:-1]) / (q[1:] - q[0:-1])
#    print("taoq[1:]:", taoq[1:], "taoq[0:-1]:", taoq[0:-1], "q[1:]:", q[1:], "q[0:-1]:", q[0:-1])
#    d2_taoq = (d1_taoq[1:] - d1_taoq[0:-1]) / (q[1:-1] - q[0:-2])
#    # print("d1_taoq[1:]:",d1_taoq[1:],"d1_taoq[0:-1]:",d1_taoq[0:-1],"q[1:-1]:",q[1:-1],"q[0:-2]:",q[0:-2])
#    # print(d2_taoq)
#    for i in range(0, len(d2_taoq)):
#        if (q[i + 1] >= 1):
#            print("i:", i, "q[i+1]:", q[i + 1], "q[i]:", q[i])
#            sigma2 = d2_taoq[i] / (d * np.log(b))
#            beta = 1 + d1_taoq[i] / d - sigma2 * (np.log(b) / 2) * (2 * q[i + 1] - 1)
#            # beta=1+d1_taoq[i]/d+sigma2*(np.log(b)/2)*(2*q[i+1]-1)
#            print("d1_taoq[i]:", d1_taoq[i])
#            break
#    # beta=0
#    print("beta:", beta, "sigma^2:", sigma2)
#    return beta, sigma2

def lognormalBetaFit(q, taoq):
    d = 2
    b = 2
    # normal_sigma2 = 2.60649  #20090105pdfN
        # 2.86059  #20090105
    # 2.77499

    d1_taoq = (taoq[1:] - taoq[0:-1]) / (q[1:] - q[0:-1])
    print("taoq[1:]:", taoq[1:], "taoq[0:-1]:", taoq[0:-1], "q[1:]:", q[1:], "q[0:-1]:", q[0:-1])
    d2_taoq = (d1_taoq[1:] - d1_taoq[0:-1]) / (q[1:-1] - q[0:-2])
    print("d1_taoq[1:]:",d1_taoq[1:],"d1_taoq[0:-1]:",d1_taoq[0:-1],"q[1:-1]:",q[1:-1],"q[0:-2]:",q[0:-2])
    print(d2_taoq)
    for i in range(0, len(d2_taoq)):
        if (q[i + 1] >= 1):
            print("i:", i, "q[i+1]:", q[i + 1], "q[i]:", q[i])
            #gamma�ֲ�
            #sigma2 = (d2_taoq[i] * q[i+1]**2 * gamma_beta) / (d2_taoq[i] * q[i+1]**2 * np.log(b) - d*gamma_alpha)
            #print('sigma2:',sigma2)
            #beta = 1 + d1_taoq[i] / d - gamma_alpha * math.log((1 - sigma2 * math.log(b)),2) / gamma_beta - (sigma2 * gamma_alpha)/(q[i+1]*gamma_beta - q[i+1]*sigma2*math.log(b))
            # ��׼��̬�ֲ�
            sigma2 = d2_taoq[i] / (d * np.log(b))
            beta=1+d1_taoq[i]/d+sigma2*(np.log(b)/2)*(2*q[i+1]-1)
            # print('sigma2:', sigma2)
            #��̬�ֲ�
            # sigma2 = d2_taoq[i] / (d * normal_sigma2 * np.log(b))
            # beta = 1 + d1_taoq[i]/d + d2_taoq[i] / 2*d - d2_taoq[i]*q[i+1] / d
            # print("d1_taoq[i]:", d1_taoq[i])
            break
    # beta=0
    print("beta:", beta, "sigma^2:", sigma2)
    return beta, sigma2

"����beta��sigma��Ȩ��W���������������,��0.25��Ľ�ˮ�����߶ȵ�0.01��"


#def mfDownscaleField(field, beta, sigma2):
#    # n=4#8*8��
#    n = 6  # 32*32
#    b = 2
#    fieldMatrix = np.array([])
#    cascade = []
#    for i in range(0, n + 1):
#        cascade.append(np.empty((b ** i, b ** i), np.float))
#    temp1 = beta - sigma2 * np.log(2) / 2
#    temp2 = np.sqrt(sigma2)
#    # fieldN=normalize(field)#
#
#    simfield = np.empty((32, 32))
#    for i in range(field.shape[0]):
#        for j in range(field.shape[1]):
#            cascade[0][0][0] = field[i, j]
#            for x in range(1, n + 1):
#                for y in range(0, b ** (x - 1)):
#                    for z in range(0, b ** (x - 1)):
#                        cascade[x][y * 2][z * 2] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(0, 1) - 2)
#                        cascade[x][y * 2][z * 2 + 1] = cascade[x - 1][y][z] * b ** (
#                        temp1 + temp2 * random.gauss(0, 1) - 2)
#                        cascade[x][y * 2 + 1][z * 2] = cascade[x - 1][y][z] * b ** (
#                        temp1 + temp2 * random.gauss(0, 1) - 2)
#                        cascade[x][y * 2 + 1][z * 2 + 1] = cascade[x - 1][y][z] * b ** (
#                        temp1 + temp2 * random.gauss(0, 1) - 2)
#            simfield[:, :] = coarseGrain(cascade[n], (32, 32))
#            # print("simfield:",simfield)
#            if (j == 0):
#                fieldRow = simfield.copy()
#            else:
#                fieldRow = np.hstack((fieldRow, simfield.copy()))
#        if (i == 0):
#            fieldMatrix = fieldRow.copy()
#        else:
#            fieldMatrix = np.vstack((fieldMatrix, fieldRow.copy()))
#   # np.savetxt('C:/Users/xia/Desktop/20060101vs20060622vs20061025/0607_1007/'+str(2006072010073030)+'_'+str(2)+'.'+"txt",fieldMatrix,fmt = '%.8f')
#    return (fieldMatrix)

def mfDownscaleField(field, beta, sigma2):
    # n=4#8*8��
    n = 6  # 32*32
    b = 2

    normal_sigma2 =1
        # 2.60649  #20090105pdfN
        # 2.86059 #20090105
        # 2.77499  #20100708trmmpdf
        # 2.81112  #20100708pdf

    normal_sigma = np.sqrt(normal_sigma2)
    normal_mu = 0
        # 0.23314  #20090105pdfN
        # 0.130297  #20090105
        # 2.57828  #20100708trmmpdf
        # 2.80668  #20100708

    fieldMatrix = np.array([])
    cascade = []
    for i in range(0, n + 1):
        cascade.append(np.empty((b ** i, b ** i), np.double))

    #��׼��̬�ֲ�
    temp2 = np.sqrt(sigma2)
    temp1 = beta - sigma2 * np.log(2) / 2

    #gamma�ֲ�
    #temp1 = beta + gamma_alpha * math.log((1-sigma2*math.log(b)),2) / gamma_beta
    #temp2 = sigma2
    #print('temp2,temp1:',temp2,temp1)

    #��̬�ֲ�
    # temp2 = np.sqrt(sigma2)
    # temp1 = beta - temp2*normal_mu - normal_sigma2*sigma2*np.log(b) / 2

    simfield = np.empty((32, 32))
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            cascade[0][0][0] = field[i, j]
            for x in range(1, n + 1):
                for y in range(0, b ** (x - 1)):
                    for z in range(0, b ** (x - 1)):
                        cascade[x][y * 2][z * 2] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(normal_mu,normal_sigma))
                        cascade[x][y * 2][z * 2 + 1] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(normal_mu,normal_sigma))
                        cascade[x][y * 2 + 1][z * 2] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(normal_mu,normal_sigma))
                        cascade[x][y * 2 + 1][z * 2 + 1] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(normal_mu,normal_sigma))
                        #cascade[x][y * 2][z * 2] = cascade[x - 1][y][z] * b **(temp1 + temp2 * random.lognormvariate(0.102470841488,0.897529158512))
                        #cascade[x][y * 2][z * 2 + 1] = cascade[x - 1][y][z] * b **(temp1 + temp2 * random.lognormvariate(0.102470841488,0.897529158512))
                        #cascade[x][y * 2 + 1][z * 2] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.lognormvariate(0.102470841488,0.897529158512))
                        #cascade[x][y * 2 + 1][z * 2 + 1] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.lognormvariate(0.102470841488,0.897529158512))
            simfield[:, :] = coarseGrain(cascade[n], (32, 32))
            # print("simfield:",simfield)
            if (j == 0):
                fieldRow = simfield.copy()
            else:
                fieldRow = np.hstack((fieldRow, simfield.copy()))
        if (i == 0):
            fieldMatrix = fieldRow.copy()
        else:
            fieldMatrix = np.vstack((fieldMatrix, fieldRow.copy()))
    # np.savetxt('C:/Users/xia/Desktop/20060101vs20060622vs20061025/0607_1007/'+str(2006072010073030)+'_'+str(2)+'.'+"txt",fieldMatrix,fmt = '%.8f')
    return (fieldMatrix)

#"��������"
def dataImport(tag):

    #"��������"
    rawIMERG = np.loadtxt('C:/Users/xia/Desktop/2006_2010/200901/20090105_trmm_3232.txt')
    assert len(rawIMERG)!=srows*scols,"���ݷ�Χ����ȷ��"
    IMERG= np.array(rawIMERG).reshape(srows,scols)
    for i in range(0, srows):
        for j in range(0, scols):
            if (IMERG[i,j]<0):
                IMERG[i, j]=0
            else:
                IMERG[i, j]=IMERG[i,j]
                
    rawAveIMERG = np.loadtxt('C:/User/200601_201001_LTDM_DEM_NDVIMin_025.txt')
    assert len(rawAveIMERG) != srows * scols, "���ݷ�Χ����ȷ��"
    AveIMERG= np.array(rawAveIMERG).reshape(srows,scols)
    
    #"���ʻ�"
    detrendData= np.empty((srows,scols),np.double)
    for i in range(0, srows):
        for j in range(0, scols):
            if (AveIMERG[1][i, j] > 0):
                detrendData[i, j] = IMERG[i, j] / AveIMERG[i, j]
            else:
                detrendData[i, j] = 0
    # "�ָ�������ѡ�ú�������"
    if tag=='MFn-GWR':
        rawGWR = np.loadtxt( 'C:/Users/hb/G_200601_201001_dem_slope_ltd_ltn_32.txt')
        assert len(rawGWR) != erows * ecols, "���ݷ�Χ����ȷ��"
        GWR = np.array(rawGWR).reshape(erows,ecols)
        for i in range(0, erows):
            for j in range(0, ecols):
                if (GWR[i, j] < 0):
                    GWR[i, j] = 0
                else:
                    GWR[i, j] = GWR[i, j]
        return detrendData,GWR
    else:
        return detrendData,AveIMERG

#"ͨ������ͳ�ƾأ����Ͻ��ж��ط�����������������ͼ�����beta��sigma"
def mutifractalAnalysis(rawfield, branch=2):

    #"��һ��"
    field = normalize(rawfield)
    # field=rawfield
    fieldSize = field.shape[0]
    layers = np.arange(0, int(math.log(fieldSize, branch)))
    scales = branch ** layers
    print("layers:", layers, "scales:", scales)
    q = np.linspace(-5, 5, 11)
    moment = np.zeros((len(layers), len(q)))

    "��moment"
    for i in range(0, len(layers)):
        for j in range(0, len(q)):
            distrib = coarseGrain(field, [x // scales[i] for x in field.shape])
            positiveDist = distrib[distrib > 0]
            # sum_positiveDist=np.sum(positiveDist)
            moment[i, j] = np.sum(positiveDist ** q[j])
            # print("distrib",distrib,"q[j]",q[j],"moment[i,j]",moment[i,j])

    "��tao(q),tao(q)����б��"
    k1 = np.zeros(moment.shape[1])  # ���б��
    b1 = np.zeros(moment.shape[1])  # ��Žؾ�
    R1_Squared = np.zeros(moment.shape[1])  # ���R2
    # k2=np.zeros(positiveDist[1])
    # b2=np.zeros(positiveDist[1])
    # R2_Squared=np.zeros(moment.shape[1])
    lambd = scales
    X = np.log(lambd) / np.log(2)  # log��2Ϊ�׵�lambda,������С���˵�X����
    X = sm.add_constant(X.T)  # ���Ͻؾ���
    # for i in range(0,positiveDist.shape[1]):
    #        Y2=np.log(positiveDist[:,i])/np.log(2)
    #        results2=sm.OLS(Y2,X).fit()
    #        k2[i]=results2.params[1]
    #        b2[i]=results2.params[0]
    #        R2_Squared[i]=results2.rsquared
    #        print("k2",k2[i],"b2",b2[i],"r22",R2_Squared[i])
    for i in range(0, moment.shape[1]):
        Y1 = np.log(moment[:, i]) / np.log(2)  # log��2Ϊ�׵�moment��������С���˵�X����
        results1 = sm.OLS(Y1, X).fit()  # log��2Ϊ�׵�moment��lambda���������
        k1[i] = results1.params[1]  # б��
        b1[i] = results1.params[0]  # �ؾ�
        R1_Squared[i] = results1.rsquared
        print("k1", k1[i], "b1", b1[i], "r21", R1_Squared[i])

    plotMoment_lambd(scales, q, k1, b1, moment, R1_Squared)
    plotD_q(k1, q)
    return (scales, q, k1, b1, moment, R1_Squared)
def main():

    global res,ratio,srows,scols,erows,ecols
    res= 0.1
    ratio = 4
    srows = 96
    scols = 96
    erows=srows*ratio
    ecols=scols*ratio

    tag='MF'#'MFn' 'MFn-GWR'
    rainfall,toRetrend=dataImport(tag)
    print (rainfall,toRetrend)
    #beta,sigma=mutifractalAnalysis(rainfall)

main()



def main2():
    Field = {}
    #Field_1D={}
    Sum_Field = 0.0
    Ave_Field = 0.0
#    Field[10] = np.loadtxt(
#        'C:/Users/xia/Desktop/��ʵ��/2007/200702100_gamma_no_zero_hb_n/2007023232_mf10.txt')
#    Field_1D[10] = np.array(Field[10]).reshape(1048576, 1)
    for count in range(1, 21):
        Field[count] = np.loadtxt( 'C:/Users/xia/Desktop/2006_2010/200901/pdf_downscaling/20090105us/' + str(200901053232)+'_pdf_gaussUSigmag_Gmf'+str(count)+ '.' + "txt")
        #Field_1D[count] = np.array(Field[count]).reshape(1048576, 1)
        Sum_Field += Field[count]
        Ave_Field = Sum_Field / 20
    np.savetxt('C:/Users/xia/Desktop/2006_2010/200901/pdf_downscaling/20090105us/' + str(200901053232) + '_pdf_gaussUSigmag_Gmf_20Ave'+ '.txt',Ave_Field, fmt='%.8f')
#main2()

def main3():
    res = 0.25
    "���ݴ���"
    trmm3b43_hb_ID = {}
    trmm3b43_hb_ID = np.loadtxt('C:/Users/xia/Desktop/2006_2010/200901/20070228_trmm_3232.txt')
    #trmm3b43_3232 = {}
    #trmm3b43_3232=np.zeros((1024,1),np.double)
    #for i in range(0,len(trmm3b43_hb_ID)):
    #    trmm3b43_3232[int(trmm3b43_hb_ID[i][0])]=trmm3b43_hb_ID[i][1]

    trmm3b43 = {}

    trmm3b43[1998] = np.array(trmm3b43_hb_ID).reshape(32, 32)

    trmm3b43NormAver_1D = {}
    trmm3b43NormAver_1D = np.loadtxt(
        'C:/Users/xia/Desktop/��ʵ��/200602_201002_Avg/200602_201002_LTNA_DEM_NDVIMax_025.txt')
    trmm3b43NormAver = {}
    trmm3b43NormAver[1] = np.array(trmm3b43NormAver_1D).reshape(32, 32)
    GWR_1D = {}
    GWR_1D = np.loadtxt(
        'C:/Users/xia/Desktop/��ʵ��/200602_201002_Avg/GWR/GWR_LTNA_DEM_NDVIMax/200602_201002_LTNA_DEM_NDVIMax_00078125_Z.txt')
    GWR = {}
    GWR[1] = np.array(GWR_1D).reshape(1024, 1024)
    for i in range(0, 1024):
        for j in range(0, 1024):
            if (GWR[1][i, j] < 0):
                GWR[1][i, j] = 0
            else:
                GWR[1][i, j] = GWR[1][i, j]

    trmm3b43Detrend = {}
    trmm3b43Detrend[199801] = np.empty((32, 32), np.double)
    for i in range(0, 32):
        for j in range(0, 32):
            if (trmm3b43NormAver[1][i, j] > 0):
                trmm3b43Detrend[199801][i, j] = trmm3b43[1998][i, j] / trmm3b43NormAver[1][i, j]
            else:
                trmm3b43Detrend[199801][i, j] = 0

    "���ط��ι���"
    print("trmm3b43Detrend:", trmm3b43Detrend[199801])
    combinedField_1D = {}
    combinedField = np.empty((32 * 32, 32 * 32), np.double)
    a, f = legendre(trmm3b43Detrend[199801])
    # a,f=legendre(trmm3b43[1998])
    # scales,q,k,b,moment,R_Squared=mutifractalAnalysis(trmm3b43[1998])
    scales, q, k, b, moment, R_Squared = mutifractalAnalysis(trmm3b43Detrend[199801])

    taoq = -k
    beta, sigma2 = lognormalBetaFit(q, taoq)
    print("taoq:", taoq, "q:", q)

    detrendField = mfDownscaleField(trmm3b43Detrend[199801], beta, sigma2)

    # detrendField=mfDownscaleField(trmm3b43[1998],beta,sigma2)
    "ѭ��100��"
    #        for count in range(1,101):
    #                detrendField=mfDownscaleField(trmm3b43Detrend[199801],beta,sigma2)
    #                for i in range(0,32):
    #                        for j in range(0,32):
    #                                temp1=trmm3b43NormAver[1][i,j]*detrendField[i*30:(i+1)*30,j*30:(j+1)*30]
    #                                temp2=temp1/np.sum(temp1)*30**2
    #                                combinedField[i*30:(i+1)*30,j*30:(j+1)*30]=temp2*trmm3b43[1998][i,j]
    #                combinedField_1D=np.array(combinedField).reshape(921600,1)
    #                np.savetxt('C:/Users/xia/Desktop/��ʵ��/201007/'+str(2010073030)+'_Fmf_1D'+'.'+"txt",combinedField_1D,fmt = '%.8f')
    #                #np.savetxt('C:/Users/xia/Desktop/20060101vs20060622vs20061025/201007/100/'+str(2010073030)+'_'+str(count)+'.'+"txt",combinedField,fmt = '%.8f')


    "û����trmm3b43Detrend[199801]"

    for i in range(0, 32):
        for j in range(0, 32):
            temp1 = detrendField[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] * np.sum(trmm3b43Detrend[199801])
            temp2 = trmm3b43NormAver[1][i, j] * temp1
            #temp1 = detrendField[i * 32:(i + 1) * 32,j * 32:(j + 1) * 32] *  np.sum(trmm3b43Detrend[199801])
            #temp2 = GWR[1][i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] * temp1

            if (np.sum(temp2) != 0):
                temp3 = temp2 / np.sum(temp2)
            else:
                temp3 = 0
            # combinedField[i*25:(i+1)*25,j*25:(j+1)*25]=temp1*trmm3b43[1998][i,j]
            combinedField[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = temp3 * (32 ** 2)
    combinedField_1D = np.array(combinedField).reshape(1048576, 1)
    np.savetxt('C:/Users/xia/Desktop/��ʵ��/200702/' + str(200702283232) + '_gamma_no_zero_Nmf_new' + '.' + "txt",combinedField_1D,fmt='%.8f')
def main1():
    trmm3b43 = {}
    trmm3b43[1998] = np.loadtxt('C:/Users/xia/Desktop/2006_2010/200901/200901053232.txt')

    trmm3b43NormAver = {}
    trmm3b43NormAver[1] = np.loadtxt(
        'C:/Users/xia/Desktop/20060101vs20060622vs20061025/20080710/0607_1007_025AVE_R.txt')

    GWR = {}
    GWR[1] = np.loadtxt(
        'C:/Users/xia/Desktop/��ʵ��/200601_201001_Avg/GWR/LTDM_DEM_NDVIMin/200601_201001_LTDM_DEM_NDVIMin_00078125_Z.txt')
    detrendField = np.empty((32 * 32, 32 * 32))
    detrendGWR = np.empty((32 * 32, 32 * 32))
    combinedField = np.loadtxt('C:/Users/xia/Desktop/20060101vs20060622vs20061025/20080710/200807103030.txt')
    for i in range(0, 32):
        for j in range(0, 32):
            if (trmm3b43[1998][i, j] == 0):
                temp1 = 0
            else:
                temp1 = combinedField[i * 30:(i + 1) * 30, j * 30:(j + 1) * 30] / trmm3b43[1998][i, j]
            if (trmm3b43NormAver[1][i, j] == 0):
                detrendField[i * 30:(i + 1) * 30, j * 30:(j + 1) * 30] = 0
            else:
                detrendField[i * 30:(i + 1) * 30, j * 30:(j + 1) * 30] = temp1 / trmm3b43NormAver[1][i, j]
            temp2 = detrendField[i * 30:(i + 1) * 30, j * 30:(j + 1) * 30] * GWR[1][i * 30:(i + 1) * 30,
                                                                             j * 30:(j + 1) * 30]
            temp3 = temp2 / np.sum(temp2)  # normalizing
            detrendGWR[i * 30:(i + 1) * 30, j * 30:(j + 1) * 30] = temp3 * trmm3b43[1998][
                i, j] * 30 ** 2  # large scale forcing

    np.savetxt('C:/Users/xia/Desktop/20060101vs20060622vs20061025/20080710/' + str(
        200807103030) + '_' + 'combine' + '.' + "txt", detrendGWR, fmt='%.8f')

