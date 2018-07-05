import numpy as np
import math
import sympy

# v=np.sum((d-e)*(d-e))/9216
# print (v)
#
# def coarseGraining(field, coarseShape):
#     # "计算聚合时的窗口大小"
#     rowRatio = sympy.S(field.shape[0]) / coarseShape[0]
#     colRatio = sympy.S(field.shape[1]) / coarseShape[1]
#     # print rowRatio
#     # "循环计算当前层级每个格网的取值"
#     # "针对非整数倍的粗粒化（类似插值但不是），按占原整格面积的比例进行累加。"
#     # "制作的window其实是一个面积比例，window与field相乘其实就只是按位置相乘"
#     # "由于i,j是先行后列，所以我对h和v进行了调整，也先行后列，然后由于field[math.floor(bottom):math.ceil(top),math.floor(left):math.ceil(right)]是第一个参数为行，"
#     # "后一个参数为列，所以两个参数的顺序也得调整"
#     # "看似只不过减少了一行转置操作，实则在window的理解和field的循环上都更清楚了，先行后列！"
#     coarseField = np.zeros((coarseShape), np.float)
#
#     top = 0
#     for i in range(0, coarseShape[0]):
#
#         bottom = top
#         top = bottom + colRatio
#         window_v = np.zeros((math.ceil(top) - math.floor(bottom)), np.float)
#         for k in range(int(math.floor(bottom)), int(math.ceil(top))):
#             if (k == math.floor(bottom)):
#                 window_v[k - math.floor(bottom)] = math.floor(bottom) + 1 - bottom
#             elif (k == math.ceil(top) - 1):
#                 window_v[k - math.floor(bottom)] = top + 1 - math.ceil(top)
#             else:
#                 window_v[k - math.floor(bottom)] = 1
#         window_v.shape = len(window_v), 1
#         # print(window_v)
#
#         right = 0
#         for j in range(0, coarseShape[1]):
#             left = right
#             right = left + rowRatio
#             window_h = np.zeros((math.ceil(right) - math.floor(left)), np.float)  #
#             for k in range(int(math.floor(left)), int(math.ceil(right))):
#                 if (k == math.floor(left)):
#                     window_h[k - math.floor(left)] = math.floor(left) + 1 - left
#                 elif (k == math.ceil(right) - 1):
#                     window_h[k - math.floor(left)] = right + 1 - math.ceil(right)
#                 else:
#                     window_h[k - math.floor(left)] = 1
#             window_h.shape = 1, len(window_h)
#             # print(window_h)
#
#             window = window_v * window_h
#             # print window
#             # window = np.transpose(window)
#             # 对于数组的相乘，“*”号意思是对应相乘，对于矩阵来说才是矩阵相乘。
#             coarseField[i, j] = np.sum(
#                 window * field[math.floor(bottom):math.ceil(top), math.floor(left):math.ceil(right)])
#             # print(coarseField[i, j])
#     return coarseField
# f=((1,2,3),(4,5,6),(7,8,9))
# g=coarseGraining(f,(2,2))
# print (g)
#
# def normalize(a):
#     min=999999.0
#     max=-min
#     for i in range(0,len(a)):
#         if (a[i]<min):
#             min=a[i]
#         if (a[i]>max):
#             max=a[i]
#     b=[]
#     for i in range(0,len(a)):
#         b[i]=(a[i]-min)/(max-min)
#     return b
#
#     b=np.array(a).reshape(len(a),1)

