# coding:utf-8
import urllib2
import re
import csv
from datetime import timedelta, datetime

yesterday = datetime.today() + timedelta(-1)
# date ='2018-03-25'
date = yesterday.strftime('%Y-%m-%d')
# print yesterday_format #2015-01-05
# date = '2018-11-11'
file=open("F:\\Data\\PD\\CSV\\" + date + ".csv", 'wb')
pre = csv.writer(file, dialect='excel')
RGS = ['StationName=%CE%E4%BA%BA%CA%D0%C7%F8&datatime=' + date, 'StationName=%B2%CC%B5%E9%C7%F8&datatime=' + date,
       'StationName=%BD%AD%CF%C4%C7%F8&datatime=' + date, 'StationName=%BB%C6%DA%E9%C7%F8&datatime=' + date,
       'StationName=%D0%C2%D6%DE%C7%F8&datatime=' + date,
       'StationName=%D2%CB%B2%FD%CA%D0%C7%F8&datatime=' + date, 'StationName=%B3%A4%D1%F4%CF%D8&datatime=' + date,
       'StationName=%B5%B1%D1%F4%CA%D0&datatime=' + date, 'StationName=%CE%E5%B7%E5%CF%D8&datatime=' + date,
       'StationName=%D0%CB%C9%BD%CF%D8&datatime=' + date, 'StationName=%D2%CB%B6%BC%CA%D0&datatime=' + date,
       'StationName=%D4%B6%B0%B2%CF%D8&datatime=' + date, 'StationName=%D6%A6%BD%AD%CA%D0&datatime=' + date,
       'StationName=%EF%F6%B9%E9%CF%D8&datatime=' + date,
       'StationName=%BE%A3%D6%DD%CA%D0%C7%F8&datatime=' + date, 'StationName=%B9%AB%B0%B2%CF%D8&datatime=' + date,
       'StationName=%BA%E9%BA%FE%CA%D0&datatime=' + date, 'StationName=%BC%E0%C0%FB%CF%D8&datatime=' + date,
       'StationName=%CA%AF%CA%D7%CA%D0&datatime=' + date, 'StationName=%CB%C9%D7%CC%CA%D0&datatime=' + date,
       'StationName=%CF%E5%D1%F4%CA%D0%C7%F8&datatime=' + date, 'StationName=%B1%A3%BF%B5%CF%D8&datatime=' + date,
       'StationName=%B9%C8%B3%C7%CF%D8&datatime=' + date, 'StationName=%C0%CF%BA%D3%BF%DA%CA%D0&datatime=' + date,
       'StationName=%C4%CF%D5%C4%CF%D8&datatime=' + date, 'StationName=%D2%CB%B3%C7%CA%D0&datatime=' + date,
       'StationName=%D4%E6%D1%F4%CA%D0&datatime=' + date,
       'StationName=%BB%C6%CA%AF%CA%D0%C7%F8&datatime=' + date, 'StationName=%B4%F3%D2%B1%CA%D0&datatime=' + date,
       'StationName=%D1%F4%D0%C2%CF%D8&datatime=' + date,
       'StationName=%BE%A3%C3%C5%CA%D0%C7%F8&datatime=' + date, 'StationName=%BE%A9%C9%BD%CF%D8&datatime=' + date,
       'StationName=%D6%D3%CF%E9%CA%D0&datatime=' + date,
       'StationName=%BB%C6%B8%D4%CA%D0%C7%F8&datatime=' + date, 'StationName=%BA%EC%B0%B2%CF%D8&datatime=' + date,
       'StationName=%BB%C6%C3%B7%CF%D8&datatime=' + date, 'StationName=%C2%DE%CC%EF%CF%D8&datatime=' + date,
       'StationName=%C2%E9%B3%C7%CA%D0&datatime=' + date, 'StationName=%CE%E4%D1%A8%CA%D0&datatime=' + date,
       'StationName=%D3%A2%C9%BD%CF%D8&datatime=' + date, 'StationName=%DE%AD%B4%BA%CF%D8&datatime=' + date,
       'StationName=%E4%BB%CB%AE%CF%D8&datatime=' + date,
       'StationName=%CA%AE%D1%DF%CA%D0%C7%F8&datatime=' + date, 'StationName=%B7%BF%CF%D8&datatime=' + date,
       'StationName=%B5%A4%BD%AD%BF%DA%CA%D0&datatime=' + date, 'StationName=%D4%C7%CE%F7%CF%D8&datatime=' + date,
       'StationName=%D4%C7%CF%D8&datatime=' + date, 'StationName=%D6%F1%C9%BD%CF%D8&datatime=' + date,
       'StationName=%D6%F1%CF%AA%CF%D8&datatime=' + date,
       'StationName=%B6%F7%CA%A9%CA%D0%C7%F8&datatime=' + date, 'StationName=%B0%CD%B6%AB%CF%D8&datatime=' + date,
       'StationName=%BA%D7%B7%E5%CF%D8&datatime=' + date, 'StationName=%BD%A8%CA%BC%CF%D8&datatime=' + date,
       'StationName=%C0%B4%B7%EF%CF%D8&datatime=' + date, 'StationName=%C0%FB%B4%A8%CA%D0&datatime=' + date,
       'StationName=%CF%CC%B7%E1%CF%D8&datatime=' + date,
       'StationName=%C7%B1%BD%AD%CA%D0%C7%F8&datatime=' + date,
       'StationName=%CC%EC%C3%C5%CA%D0&datatime=' + date,
       'StationName=%CF%C9%CC%D2%CA%D0%C7%F8&datatime=' + date,
       'StationName=%CB%E6%D6%DD%CA%D0%C7%F8&datatime=' + date, 'StationName=%B9%E3%CB%AE%CA%D0&datatime=' + date,
       'StationName=%CF%CC%C4%FE%CA%D0%C7%F8&datatime=' + date, 'StationName=%B3%E0%B1%DA%CA%D0&datatime=' + date,
       'StationName=%B3%E7%D1%F4%CF%D8&datatime=' + date, 'StationName=%BC%CE%D3%E3%CF%D8&datatime=' + date,
       'StationName=%CD%A8%B3%C7%CF%D8&datatime=' + date, 'StationName=%CD%A8%C9%BD%CF%D8&datatime=' + date,
       'StationName=%D0%A2%B8%D0%CA%D0%C7%F8&datatime=' + date, 'StationName=%B0%B2%C2%BD%CA%D0&datatime=' + date,
       'StationName=%B4%F3%CE%F2%CF%D8&datatime=' + date, 'StationName=%BA%BA%B4%A8%CA%D0&datatime=' + date,
       'StationName=%D3%A6%B3%C7%CA%D0&datatime=' + date, 'StationName=%D4%C6%C3%CE%CF%D8&datatime=' + date,
       'StationName=%B6%F5%D6%DD%CA%D0%C7%F8&datatime=' + date,
       'StationName=%C9%F1%C5%A9%BC%DC%C1%D6%C7%F8&datatime=' + date,
       'StationName=%CD%C5%B7%E7%CF%D8&datatime=' + date,  # 团风
       'StationName=%D0%FB%B6%F7%CF%D8&datatime=' + date,  # 宣恩
       'StationName=%D0%A2%B2%FD%CF%D8&datatime=' + date,  # 孝昌
       ]

url = 'http://www.hbqx.gov.cn/qx_tqsk.action'  # 网址

req_header = {
    'Host': 'www.hbqx.gov.cn',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    'Accept-Encoding': 'gbk, gzip, deflate',
    'Referer': 'http://www.hbqx.gov.cn/qx_tqsk.action',
    'Content-Type': 'application/x-www-form-urlencoded',
    # 'Content-Length': '50',
    # 'Cookie': 'JSESSIONID=D1F492A0F74D1287BB06D5AF2DBE58D4'
}
req_timeout = 500

for k in range(0, 78):
    req = urllib2.Request(url, RGS[k], req_header)  # RGS[k] '+date+'
    res = urllib2.urlopen(req)  # None,,req_timeout
    page = res.read()
    data = page.decode('gb2312')
    # print data
    pattern = r'data:\[.+?\]'
    t0 = re.findall(pattern, data, re.S | re.M)
    # print t0
    ulist = [999999 for i in range(0, 5)]
    for i in range(1, len(t0) - 2, 2):
        # print t0[i]
        t1 = re.split('[\[]+', t0[i])
        # print t1
        t2 = re.split('[\]]+', t1[1])
        # print t2
        t3 = re.split('[,]+', t2[0])
        # print t3
        if i == 1:
            if (len(t3) < 18):
                continue
            else:
                ui = []
                c = 0
                for j in range(0, len(t3)):
                    if (unicode(t3[j]) != 'null'):
                        ui.append(float(t3[j]))
                        c = c + 1
                if c >= 18:
                    ulist[0] = sum(ui)
        if i == 3:
            if (len(t3) < 18):
                continue
            else:
                ui = []
                c = 0
                for j in range(0, len(t3)):
                    if (unicode(t3[j]) != 'null'):
                        ui.append(float(t3[j]))
                        c = c + 1
                if c >= 18:
                    ulist[4] = sum(ui) / c
        if i == 5:
            if (len(t3) < 18):
                continue
            else:
                ui = []
                c = 0
                tmax = 0
                tmin = 100
                for j in range(0, len(t3)):
                    if (unicode(t3[j]) != 'null'):
                        ui.append(float(t3[j]))
                        c = c + 1
                        if tmax < float(t3[j]):
                            tmax = float(t3[j])
                        if tmin > float(t3[j]):
                            tmin = float(t3[j])
                if c >= 18:
                    ulist[1] = tmin
                    ulist[2] = tmax
                    ulist[3] = sum(ui) / c
    print ulist
    pre.writerow(ulist)
    print '//////////////////////////////////////'
file.close()