# coding:utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
import sys, string, os
import xlrd
import xlwt
import xlsxwriter
import numpy as np
import time
import math
import random
import sympy
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import random
arcpy.env.overwriteOutput = True

def OLS():
    arcpy.env.overwriteOutput = True
    path = 'F:/Test/Paper180829/Process/'
    for y in range(15, 18):
        for m in range(1, 13):
            name = str(((2000 + y) * 100) + m)
            shpFile = path + '10kmData/' + 'd' + name + '.shp'

            shpOut = path + '10kmData_OLSResult/' + '10kmDataOLS' + name + '.shp'
            coefOut = path + '10kmData_OLSResult/' + 'coef10kmDataOLS' + name + '.dbf'
            diagOut = path + '10kmData_OLSResult/' + 'diag10kmDataOLS' + name + '.dbf'
            arcpy.OrdinaryLeastSquares_stats(shpFile, "ORIG_FID", shpOut,
                                             "IMERG", "DEM;LTD;X;Y",
                                             coefOut, diagOut)
            print 'finish ', name
# OLS()

def ER():
    arcpy.env.overwriteOutput = True
    path = 'F:/Test/Paper180829/Process/'
    for y in range(15, 18):
        for m in range(1, 13):
            name = str(((2000 + y) * 100) + m)
            shpFile = path + '10kmData/' + 'd' + name + '.shp'

            reportOut = path + '10kmData4Var_ERResult/' + 'report10kmData4VarER' + name + '.txt'
            resultOut = path + '10kmData4Var_ERResult/' + 'result10kmData4VarER' + name + '.dbf'
            arcpy.ExploratoryRegression_stats(shpFile,
                                              "IMERG",
                                              "DEM;LTD;X;Y",
                                              "",reportOut, resultOut)
            print 'finish ', name
# ER()

