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


def f1():
    pathIn = 'F:/Test/Data/IMERG/' + 'IMERGM_nc_201501_201805_HB91_51/'
    pathOut = 'F:/Test/Data/IMERG/' + 'IMERGM_tif_201501_201805_HB91_51/'
    arcpy.env.workspace = pathIn
    for nc_file in arcpy.ListFiles("*.nc"):
        time = nc_file[20:28]
        y, m, d = int(time[0:4]), int(time[4:6]), int(time[6:8])
        layer = 'nc_' + time
        arcpy.MakeNetCDFRasterLayer_md(nc_file, "precipitation", "lon", "lat", layer)  # "nc制作图层"
        if (m == 2):
            if (y == 2016):
                times = 29 * 24
            else:
                times = 28 * 24
        elif (m == 4 or m == 6 or m == 9 or m == 11):
            times = 30 * 24
        else:
            times = 31 * 24
        print y, m, d
        outTimes = Times(layer, times)
        outTimes.save(pathOut + 'I' + time[0:6] + '.tif')
