# coding:utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

#"像元聚合,粗粒化：coarseGraining"
def coarseGraining(field, coarseShape):

    # "计算聚合时的窗口大小"
    rowRatio = float(field.shape[0]) / coarseShape[0]
    colRatio = float(field.shape[1])/ coarseShape[1]
    #print rowRatio
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
                window * field[math.floor(bottom):math.ceil(top),math.floor(left):math.ceil(right)] )
            #print(coarseField[i, j])
    return coarseField


#多重分形特征分析mutifractalAnalysis
def mutifractalAnalysis(field, branch=2):

    #"首先将消除趋势波动（Detrended Fluctuation）的降水场归一化"
    sumField=np.sum(field)
    if (sumField > 0):
        field=field/sumField
    # print field

    fieldSize = field.shape[0]
    # "layers+1 即向上分析的层数，scales即每层中像元大小对应的起始0.1度时的倍数"
    layers = np.arange(0, int(math.log(fieldSize, branch)))
    scales = branch ** layers
    print("layers:", layers, "scales:", scales)
    q = np.linspace(-5, 5, 11)

    #"求统计矩moment"
    moment = np.zeros((len(layers), len(q)))
    for i in range(0, len(layers)):
        for j in range(0, len(q)):
            # print field.shape // scales[i]
            distrib = coarseGraining(field, field.shape//scales[i])     ##[x // scales[i] for x infield.shape]
            positiveDist = distrib[distrib > 0]
            moment[i, j] = np.sum(positiveDist ** q[j])
            #print"distrib",distrib
            #print "q[j]",q[j]
            #print "moment[i,j]",moment[i,j]

    #"求tao(q),tao(q)就是斜率"
    k = np.zeros(moment.shape[1])  # 存放斜率
    b = np.zeros(moment.shape[1])  # 存放截距
    R_Squared = np.zeros(moment.shape[1])  # 存放R方
    lambd = scales
    X = np.log(lambd) / np.log(2)  # log以2为底的lambda,线性最小二乘的X输入
    for i in range(0, moment.shape[1]):
        Y1 = np.log(moment[:, i]) / np.log(2)  # log以2为底的moment，线性最小二乘的X输入
        results1 = sm.OLS(Y1, X).fit()  # log以2为底的moment和lambda的线性拟合
        k1[i] = results1.params[1]  # 斜率
        b1[i] = results1.params[0]  # 截距
        R1_Squared[i] = results1.rsquared
        print("k1", k1[i], "b1", b1[i], "r21", R1_Squared[i])

    plotMoment_lambd(scales, q, k1, b1, moment, R1_Squared)
    plotD_q(k1, q)
    return (scales, q, k1, b1, moment, R1_Squared)


def main():
    
    #"数据处理"
    data = {}
    data = np.loadtxt('F:/Test/MFn/IM201607.txt') #IM201607
    IMERG = {}#原始降水数据
    IMERG= np.array(data).reshape(96, 96)
    for i in range(0, 96):
        for j in range(0, 96):
            if (IMERG[i, j] < 0):
                IMERG[i, j] = 0
            else:
                IMERG[i, j] = IMERG[i, j]
#    print IMERG
    data = {}
    data = np.loadtxt('F:/Test/MFn/3IM.txt')
    AveIMERG = {}#用来匀质化的平均数据
    AveIMERG = np.array(data).reshape(96, 96)
    for i in range(0, 96):
        for j in range(0, 96):
            if (AveIMERG[i, j] < 0):
                AveIMERG[i, j] = 0
            else:
                AveIMERG[i, j] = AveIMERG[i, j]
    # data = {}
    # data = np.loadtxt('F:/Test/')
    # GWR = {}#恢复异质性的GWR结果
    # GWR= np.array(data).reshape(96, 96)
    # for i in range(0, 96):
    #     for j in range(0, 96):
    #         if (GWR[i, j] < 0):
    #             GWR[i, j] = 0
    #         else:
    #             GWR[i, j] = GWR[i, j]
    
    #"匀质化"
    DeIMERG = {}
    DeIMERG= np.empty((96, 96),np.double)
    for i in range(0, 96):
        for j in range(0, 96):
            if (AveIMERG[i, j] > 0):
                DeIMERG[i, j] = IMERG[i, j] / AveIMERG[i, j]
            else:
                DeIMERG[i, j] = 0
#    print("DeIMERG:", DeIMERG)

    #"多重分形特征验证"

    #"从32*32到1*1，用往上聚合的方法，以广义分型维度法D(q)，确认具有分形特征"
    scales, q, k, b, moment, R_Squared = mutifractalAnalysis(DeIMERG)

    "计算奇异值α和多重分形谱f（α），实际还是用统计矩计算广义分型维度，再以勒让德变化得出的，确认具有分形特征"
    a, f = legendre(trmm3b43Detrend[199801])
    # a,f=legendre(trmm3b43[1998])
    # scales,q,k,b,moment,R_Squared=mutifractalAnalysis(trmm3b43[1998])

    "前面都只是确认具有分形特征，所以从32*32聚合计算到1*1，"
    "q取-5到5是为了确认具有多重分形特征，即函数为单调凸函数"

    combinedField_1D = {}
    combinedField = np.empty((32 * 32, 32 * 32),np.double)

main()