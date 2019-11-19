# coding:gbk

#
#
# def f1():
#     pathIn = 'F:/Test/Data/IMERG/' + 'IMERGM_nc_201501_201805_HB91_51/'
#     pathOut = 'F:/Test/Data/IMERG/' + 'IMERGM_tif_201501_201805_HB91_51/'
#     arcpy.env.workspace = pathIn
#     for nc_file in arcpy.ListFiles("*.nc"):
#         time = nc_file[20:28]
#         y, m, d = int(time[0:4]), int(time[4:6]), int(time[6:8])
#         layer = 'nc_' + time
#         arcpy.MakeNetCDFRasterLayer_md(nc_file, "precipitation", "lon", "lat", layer)  # "nc����ͼ��"
#         if (m == 2):
#             if (y == 2016):
#                 times = 29 * 24
#             else:
#                 times = 28 * 24
#         elif (m == 4 or m == 6 or m == 9 or m == 11):
#             times = 30 * 24
#         else:
#             times = 31 * 24
#         print y, m, d
#         outTimes = Times(layer, times)
#         outTimes.save(pathOut + 'I' + time[0:6] + '.tif')
#
# st=time.clock()
#
# for i in range(0,10000):
#     for j in range(0, 10000):
#         c=i*j
# t=time.clock()-st
# print("���߶ȼ����ʱ:{:.0f}m {:.0f}s.\n".format(int(t)/60,int(t)%60))

# def forRename(inpath,outpath):
#     for p1 in os.listdir(inpath):#���XX�أ������
#         subInpath=os.path.join(inpath,p1)
#         if os.path.isdir(subInpath)==False:# ����������ļ���������
#             shutil.copyfile(subInpath,outpath)
#
#         else:
#             subOutpath = os.path.join(outpath, p1)
#             if os.path.exists(subOutpath) == False:
#                 os.mkdir(subOutpath)
#
#             for p2 in os.listdir(subInpath):#���XX������ĳĳ���飬���������DPMXXX
#                 subSubInpath=os.path.join(subInpath,p2)
#                 if os.path.isdir(subSubInpath) == False:  # ������������ļ���������
#                     shutil.copyfile(subSubInpath, subOutpath)
#
#                 else:
#                     subSubOutpath = os.path.join(subOutpath, p2)
#                     if os.path.exists(subSubOutpath) == False:
#                         os.mkdir(subSubOutpath)
#
#                     for p3 in os.listdir(subSubInpath):#���XX������ĳĳ�����XXͼƬ�����������DPMXXX��IMG_2018_4096.img
#                         shutil.copyfile(file, subSubOutpath)
#
#                         file = os.path.join(subSubInpath, p3)
#                         name, tail = os.path.splitext(p3)[0], os.path.splitext(p3)[1]
#                         if os.path.isdir(file) == True:  # ����������ļ�����Ҳ��ȫ���ƹ�ȥ
#                             shutil.copyfile(file, subSubOutpath)
#                         elif tail== 'img' or tail=='IMG' or tail== 'jpg' or tail=='JPG'or tail== 'mp4' or tail=='MP4':
#                             partName=p3.split('_')
#                             if len(partName)<3:
#                                 newName=p3
#                             else:
#                                 newName=partName[0]
#                                 for i in range(2,len(partName)):
#                                     newName=newName+'_'+partName[i]
#
#                             outfile=os.path.join(subSubOutpath,p3)
#                             newfile=os.path.join(subSubOutpath,newName)
#                             shutil.move(outfile,newfile)
#
#                         else:
#                             shutil.copyfile(file, subSubOutpath)#������ǲ���ͼƬ������Ƶ��Ҳ��ȫ���ƹ�ȥ
#
#
#     return 0
import os,shutil
def forRename(inpath):
    for p1 in os.listdir(inpath):#���XX�أ������
        subInpath=os.path.join(inpath,p1)
        print(subInpath)
        if os.path.isdir(subInpath)==False:# ����������ļ���������
            continue
        else:
            for p2 in os.listdir(subInpath):#���XX������ĳĳ���飬���������DPMXXX
                subSubInpath=os.path.join(subInpath,p2)
                print(subSubInpath)
                if os.path.isdir(subSubInpath) == False:  # ������������ļ���������
                    file = subSubInpath
                    name, tail = os.path.splitext(p2)[0], os.path.splitext(p2)[1]
                    # print(name,tail)
                    if tail == '.img' or tail == '.IMG' or tail == '.jpg' or tail == '.JPG' or tail == '.mp4' or tail == '.MP4' or tail == '.pdf':
                        partName = p2.split('_')
                        if len(partName) < 3:
                            newName = p2
                        else:
                            newName = partName[0]
                            for i in range(2, len(partName)):
                                newName = newName + '_' + partName[i]
                        newfile = os.path.join(subInpath, newName)
                        shutil.move(file, newfile)

                else:
                    for p3 in os.listdir(subSubInpath):#���XX������ĳĳ�����XXͼƬ�����������DPMXXX��IMG_2018_4096.img
                        file = os.path.join(subSubInpath, p3)
                        print(file)
                        name, tail = os.path.splitext(p3)[0], os.path.splitext(p3)[1]
                        # print(name,tail)
                        if os.path.isdir(file) == True:  # ����������ļ�����Ҳ��ȫ���ƹ�ȥ
                            continue
                        elif tail== '.img' or tail=='.IMG' or tail== '.jpg' or tail=='.JPG'or tail== '.mp4' or tail=='.MP4' or tail=='.pdf' :
                            partName=p3.split('_')
                            if len(partName)<3:
                                newName=p3
                            else:
                                newName=partName[0]
                                for i in range(2,len(partName)):
                                    newName=newName+'_'+partName[i]

                            newfile=os.path.join(subSubInpath,newName)
                            shutil.move(file,newfile)
                        else:
                            continue
inpath="H:/��Ƭ����/2019"
forRename(inpath)