# coding:utf-8
import numpy as np
import math
import sympy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import time

# "像元聚合,粗粒化：coarseGraining"
def coarseGraining(field, coarseShape):
    # "计算聚合时的窗口大小"
    rowRatio = sympy.S(field.shape[0]) / coarseShape[0]
    colRatio = sympy.S(field.shape[1]) / coarseShape[1]
    # print rowRatio
    # "循环计算当前层级每个格网的取值"
    # "针对非整数倍的粗粒化（类似插值但不是），按占原整格面积的比例进行累加。"
    # "制作的window其实是一个面积比例，window与field相乘其实就只是按位置相乘"
    # "由于i,j是先行后列，所以我对h和v进行了调整，也先行后列，然后由于field[math.floor(bottom):math.ceil(top),math.floor(left):math.ceil(right)]是第一个参数为行，"
    # "后一个参数为列，所以两个参数的顺序也得调整"
    # "看似只不过减少了一行转置操作，实则在window的理解和field的循环上都更清楚了，先行后列！"
    coarseField = np.zeros((coarseShape), np.float)

    top = 0
    for i in range(0, coarseShape[0]):

        bottom = top
        top = bottom + colRatio
        window_v = np.zeros((math.ceil(top) - math.floor(bottom)), np.float)
        for k in range(int(math.floor(bottom)), int(math.ceil(top))):
            if (k == math.floor(bottom)):
                window_v[k - math.floor(bottom)] = math.floor(bottom) + 1 - bottom
            elif (k == math.ceil(top) - 1):
                window_v[k - math.floor(bottom)] = top + 1 - math.ceil(top)
            else:
                window_v[k - math.floor(bottom)] = 1
        window_v.shape = len(window_v), 1
        # print(window_v)

        right = 0
        for j in range(0, coarseShape[1]):
            left = right
            right = left + rowRatio
            window_h = np.zeros((math.ceil(right) - math.floor(left)), np.float)  #
            for k in range(int(math.floor(left)), int(math.ceil(right))):
                if (k == math.floor(left)):
                    window_h[k - math.floor(left)] = math.floor(left) + 1 - left
                elif (k == math.ceil(right) - 1):
                    window_h[k - math.floor(left)] = right + 1 - math.ceil(right)
                else:
                    window_h[k - math.floor(left)] = 1
            window_h.shape = 1, len(window_h)
            # print(window_h)

            window = window_v * window_h
            # print window
            # window = np.transpose(window)
            # 对于数组的相乘，“*”号意思是对应相乘，对于矩阵来说才是矩阵相乘。
            coarseField[i, j] = np.sum(
                window * field[math.floor(bottom):math.ceil(top), math.floor(left):math.ceil(right)])
            # print(coarseField[i, j])
    return coarseField

# "证明具有幂律特征"
# "作图log2统计矩与log2分型尺度"
def plotMoment_lambda(scales, q, k, b, moment, rsquared,name=""):
    plt.figure(1)
    lambd = scales
    x = np.log(lambd) / np.log(2)
    for j in range(0, moment.shape[1]):
        y = np.log(moment[:, j]) / np.log(2)
        plt.scatter(x, y)
        plt.plot(x, k[j] * x + b[j], "-")
        plt.text(x[0], y[0], 'q=' + str(q[j])[0:4] ,rotation=0)
#      plt.text(x[-3], y[-3], 'q=' + str(q[j])[0:4] + ',$R^2=$' + str(rsquared[j])[0:6],rotation=-5)  # 将q和r2的值显示在图上，以及显示的位置
    #plt.xlim(-1, 6)
    #plt.ylim(-60, 100)
    plt.xlabel(r'$Log_2[\lambda]$')
    plt.ylabel(r'$Log_2[M(\lambda,q)]$')
    plt.savefig('F:/Test/OUT/' + "MomentScale" +name+ ".png", dpi=300)
    plt.close(1)

#"证明具有多重分形特征"
#"用广义分形维数D(q)表示"
def plotD_q(q, D_q,name=""):
    plt.figure(2)
    plt.plot(q, D_q, "-o", label=name,color='blue')
    plt.plot((list(q)[0], list(q)[-1]), (list(D_q)[0], list(D_q)[-1]),color='red')
    plt.xlabel(r'$q$')
    plt.ylabel(r'$D(q)$')
    #plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.savefig('F:/Test/OUT/' + "D_q" + name+ ".png", dpi=300)
    plt.close(2)

#"证明具有多重分形特征"
#"用多重分形谱f(α)表示"
def plotAlpha_f(a, f, name=""):
    plt.figure(3)
    plt.plot(a, f, "-o", label=name,color='blue')
    plt.xlabel(r'${\alpha}$')
    plt.ylabel(r"$f(\alpha)$")
    #plt.title("多重分形谱")
    #plt.legend()
    plt.savefig('F:/Test/OUT/' + "f_alpha" + name + ".png", dpi=300)
    plt.close(3)

def plotqTaoq(q,taoq,name=""):
    plt.figure(4)
    plt.plot(q,taoq, "-o", label=name, color='blue')
    plt.plot((list(q)[0], list(q)[-1]), (list(taoq)[0], list(taoq)[-1]), color='red')
    plt.savefig('F:/Test/OUT/' + "qTaoq" +name+ ".png", dpi=300)
    plt.close(4)

def plotqAlpha(q,alpha,name=""):
    plt.figure(5)
    plt.plot(q,alpha, "-o", label=name, color='blue')
    plt.plot((list(q)[0], list(q)[-1]), (list(alpha)[0], list(alpha)[-1]), color='red')
    plt.savefig('F:/Test/OUT/' + "qAlpha" +name+ ".png", dpi=300)
    plt.close(5)

# 多重分形特征分析mutifractalAnalysis"
def mutifractalAnalysis(option=0,dimension=2,branch=2):

    s = time.clock()
    # "数据处理"
    row=96
    col=96
    data = {}
    data = np.loadtxt('F:/Test/MF2/IM9696.txt')  # IM201607
    IMERG = {}  # 原始降水数据
    IMERG = np.array(data).reshape(row,col)
    for i in range(0, row):
        for j in range(0, col):
            if (IMERG[i, j] < 0):
                IMERG[i, j] = 0
            else:
                IMERG[i, j] = IMERG[i, j]
    #    print IMERG
    data = {}
    data = np.loadtxt('F:/Test/MF2/aveIM9696.txt')
    AveIMERG = {}  # 用来匀质化的平均数据
    AveIMERG = np.array(data).reshape(row, col)
    for i in range(0, row):
        for j in range(0, col):
            if (AveIMERG[i, j] < 0):
                AveIMERG[i, j] = 0
            else:
                AveIMERG[i, j] = AveIMERG[i, j]

    # "匀质化"
    DeIMERG = {}
    DeIMERG = np.empty((row, col), np.double)
    for i in range(0, row):
        for j in range(0, col):
            if (AveIMERG[i, j] > 0):
                DeIMERG[i, j] = IMERG[i, j] / AveIMERG[i, j]
            else:
                DeIMERG[i, j] = 0
    #    print("DeIMERG:", DeIMERG)

    field=DeIMERG
    # "首先将消除趋势波动（Detrended Fluctuation）的降水场归一化"
    sumField = np.sum(field)
    if (sumField > 0):
        field = field / sumField
    # print field

    fieldSize = field.shape[0]
    # "layers+1 即向上分析的层数，scales即每层中像元大小对应的起始0.1度时的倍数"
    layers = np.arange(0, int(math.log(fieldSize, branch)))
    scales = branch ** layers
    #print("layers:", layers, "scales:", scales)
    q = np.linspace(-10, 10, 21)

    # "求统计矩moment"
    #"d1用来计算taoq的一阶导(求奇异值α和参数β")，d2用来计算taoq的二阶导（求参数σ）"
    # "d3用来计算D(q)"
    # "同时这里计算出后面需要的β和σ"

    moment = np.zeros((len(layers), len(q)))
    d1= np.zeros((len(layers), len(q)))
    d2= np.zeros((len(layers), len(q)))
    d3= np.zeros((len(layers), len(q)))
    for i in range(0, len(layers)):
        for j in range(0, len(q)):
            # print field.shape // scales[i]
            distrib = coarseGraining(field, field.shape // scales[i])  ##[x // scales[i] for x infield.shape]
            positiveDist = distrib[distrib > 0]
            moment[i, j] = np.sum(positiveDist ** q[j])
            # print"distrib",distrib
            # print "q[j]",q[j]
            # print "moment[i,j]",moment[i,j]
            qmass = positiveDist ** q[j]
            d1[i, j]=np.sum(qmass * np.log(positiveDist))/ np.sum(qmass)
            d2[i, j]=np.sum(qmass * np.log(positiveDist)**2 )/ np.sum(qmass)-d1[i,j]**2
            if (q[j]!=1):
                d3[i, j]=np.log(np.sum(qmass))/(q[j]-1)
            else:
                d3[i, j] = d1[i, j]

    # "证明具有幂律特征"# "求tao(q),tao(q)就是斜率"
    k = np.zeros(len(q))  # 存放斜率
    b = np.zeros(len(q))  # 存放截距
    R_Squared = np.zeros(len(q))  # 存放R方
    lambd = scales
    X = np.log(lambd) / np.log(2)  # log以2为底的lambda,线性最小二乘的X输入
    X = sm.add_constant(X.T)  # 加上截距项
    for i in range(0, len(q)):
        Y = np.log(moment[:, i]) / np.log(2)  # log以2为底的moment，线性最小二乘的X输入
        results = sm.OLS(Y, X).fit()  # log以2为底的moment和lambda的线性拟合
        k[i] = results.params[1]  # 斜率
        b[i] = results.params[0]  # 截距
        R_Squared[i] = results.rsquared
        #print("k:", k[i], "b:", b[i], "Rsquared:", R_Squared[i])

    # "在多重分形领域taoq就是上面的斜率，与级联降尺度中的taoq不同"
    taoq = k
    # "证明具有多重分形特征"#"用多重分形谱f(α)表示"
    alpha = np.zeros(len(q))
    f_alpha = np.zeros(len(q))
    for j in range(0, len(q)):
        line = np.polyfit(np.log(1.0 * scales), d1[:, j], 1)
        alpha[j] = line[0]
        f_alpha[j] = alpha[j] * q[j] - taoq[j]

    # "证明具有多重分形特征"# "用广义分形维数D(q)表示"
    D_q=np.zeros((len(q)))
    for j in range(0, len(q)):
        line = np.polyfit(np.log(1.0 * scales), d3[:, j], 1)
        D_q[j] = line[0]

    #"qTaoq与qAlpha"
    plotqTaoq(q,taoq,"_"+option)
    plotqAlpha(q,alpha,"_"+option)
    plotMoment_lambda(scales, q, k, b, moment, R_Squared, "_"+option)
    plotAlpha_f(alpha, f_alpha, "_"+option)
    plotD_q(q, D_q, "_"+option)

    # "求二阶导，继而计算β和σ，将q=1处作为返回值"
    # "由于级联降尺度的研究者，分形尺度仍然从1到5，强行把负号移到了taoq上，所以在Xu等的降尺度中，taoq已经变化，增加来自λ的负号"
    #scales = scales[::-1]
    #print (scales)

    temp=np.zeros(len(q))
    alpha=-alpha
    for j in range(0, len(q)):
        d2[:,j] = -d2[::,j]
        line = np.polyfit(np.log(1.0 * scales), d2[:, j], 1)
        temp[j] = line[0]
    #print (temp)

    e=0
    v=1
    for i in range(0, len(q)):
        if (q[i]>= 1):

            if (option=="MF"):
                # "X是标准正态分布"
                sigma =math.sqrt(temp[i]/(dimension*np.log(branch)))
                beta = 1 + alpha[i] /dimension  - temp[i]  * (q[i] - 0.5)/ dimension
            else:
                # "X是非标准正态分布"
                # "需要计算原数据的均值和方差"
                data = np.array(IMERG).reshape(row * col, 1)
                d = []
                for j in range(0, row * col):
                    if data[j, 0] != 0:
                        d.append(data[j,0])
                e = np.sum(np.log(d)/np.log(2)) / len(d)
                v = np.sum((np.log(d)/np.log(2) - e) ** 2) / len(d)
                #print("mu:", e, "sigma:", v)
                sigma =math.sqrt(temp[i]/(v*dimension*np.log(branch)))
                beta = 1 + alpha[i] /dimension  - temp[i]  * (q[i] - 0.5)/ dimension
            break
    print (beta,sigma,e,v)

    print("多重分形特征分析及参数计算耗时=", time.clock() - s,"秒")
    return (beta,sigma,e,v)

# "由β和σ执行降尺度"
def mfDownscaleField(option ,beta, sigma,e=0,v=1):

    st = time.clock()
    #"循环次数/级联层数/行数/列数/二分"
    m=20
    n = 4
    row=43
    col=79
    b = 2

    # "数据处理"
    data = {}
    data = np.loadtxt('F:/Test/MF2/IM7943.txt')  # IM201607
    IMERG = {}  # 原始降水数据
    IMERG = np.array(data).reshape(row,col)
    for i in range(0, row):
        for j in range(0, col):
            if (IMERG[i, j] < 0):
                IMERG[i, j] = 0
            else:
                IMERG[i, j] = IMERG[i, j]
    #    print IMERG
    data = {}
    data = np.loadtxt('F:/Test/MF2/aveIM7943.txt')
    AveIMERG = {}  # 用来匀质化的平均数据
    AveIMERG = np.array(data).reshape(row, col)
    for i in range(0, row):
        for j in range(0, col):
            if (AveIMERG[i, j] < 0):
                AveIMERG[i, j] = 0
            else:
                AveIMERG[i, j] = AveIMERG[i, j]

    data = {}
    data = np.loadtxt('F:/Test/MF2/rGWR.txt')
    cGWR = {}  # 用来匀质化的平均数据
    cGWR = np.array(data).reshape(row*2**n, col*2**n)
    for i in range(0, row*2**n):
        for j in range(0, col*2**n):
            if (cGWR[i, j] < 0):
                cGWR[i, j] = 0
            else:
                cGWR[i, j] = cGWR[i, j]

    # "匀质化"
    DeIMERG = {}
    DeIMERG = np.empty((row, col), np.double)
    for i in range(0, row):
        for j in range(0, col):
            if (AveIMERG[i, j] > 0):
                DeIMERG[i, j] = IMERG[i, j] / AveIMERG[i, j]
            else:
                DeIMERG[i, j] = 0
    #    print("DeIMERG:", DeIMERG)

    field = DeIMERG

    fieldAll=[]
    cascade = []
    for i in range(0, n + 1):
        cascade.append(np.zeros((b ** i, b ** i), np.double))
    #print ("cascade:",cascade)

    temp1 = beta -sigma * e- v * sigma ** 2 * np.log(b) / 2
    temp2 = sigma

    # "循环m次"
    for k in range(0,m):
        for i in range(row):
            for j in range(col):
                cascade[0][0][0] = field[i, j]
                for x in range(1, n + 1):
                    for y in range(0, b ** (x - 1)):
                        for z in range(0, b ** (x - 1)):
                            cascade[x][y * 2][z * 2] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(e,v))
                            cascade[x][y * 2][z * 2 + 1] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(e,v))
                            cascade[x][y * 2 + 1][z * 2] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(e,v))
                            cascade[x][y * 2 + 1][z * 2 + 1] = cascade[x - 1][y][z] * b ** (temp1 + temp2 * random.gauss(e,v))
                            .0
                #simfield[:,.  :] = coarseGraining(cascade[n], (32, 32))
                # print("simfield:",simfield)
                if (j == 0):
                    fieldRow = cascade[n].copy()
                else:
                    fieldRow = np.hstack((fieldRow, cascade[n].copy()))
            if (i == 0):
                fieldMatrix = fieldRow.copy()
            else:
                fieldMatrix = np.vstack((fieldMatrix, fieldRow.copy()))
        #np.savetxt('F:/Test/OUT/'+"fieldAll"+str(k)+""+".txt",fieldMatrix,fmt = '%.8f')
        fieldAll.append(fieldMatrix)

    # "求多次结果平均值"
    fieldAve=np.zeros((row*2**n,col*2**n),np.double)
    for k in range(0,m):
        fieldAve=fieldAve+fieldAll[k]
    fieldAve=fieldAve/m
    # np.savetxt('F:/Test/OUT/'+"fieldAve"+"ave"+".txt", fieldAve,fmt = '%.8f')


    # "恢复异质性"
    result=np.zeros((row*2**n,col*2**n),np.double)
    for i in range(0, row):
        for j in range(0, col):
            # temp=AveIMERG[i,j]*fieldAve[i*2**n:(i+1)*2**n,j*2**n:(j+1)*2**n]
            temp =cGWR[i*2**n:(i+1)*2**n,j*2**n:(j+1)*2**n]*fieldAve[i*2**n:(i+1)*2**n,j*2**n:(j+1)*2**n]
            if (np.sum(temp) != 0):
                temp = temp / np.sum(temp)
            else:
                temp = 0
            result[i*2**n:(i+1)*2**n,j*2**n:(j+1)*2**n] = temp * IMERG[i, j] * (2**n*2**n)
    result = np.array(result).reshape(row*2**n*col*2**n, 1)
    np.savetxt('F:/Test/OUT/' + "r" + option + ".txt", result, fmt='%.8f')
    print ("降尺度计算耗时=",time.clock() - st,"秒")

# 主函数"
def main():

    # "多重分形特征验证"
    # "输出5张图，并计算q=1处的β和σ，taoq相关的值统一加负号"
    # beta, sigma,e,v = mutifractalAnalysis("MFn")
    # beta  =0.0003221456447606301
    # sigma =0.08498305263749407
    # e=0
    # v=1

    beta  =0.00018610559374459146
    sigma =0.10707333803819653
    e=6.9247792607123975
    v=0.6299440380321286

    # "实际降尺度并得到结果"
    # mfDownscaleField("MFnGWRnew",beta, sigma,e,v)

main()