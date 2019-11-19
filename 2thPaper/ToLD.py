# coding:gbk
import os,shutil
def fileCopy():#
    inPath=""
    outPath=""

    for sp1 in os.listdir(inPath):# 遍历子文件夹，sp子文件夹名
        sub1inpath=os.path.join(inPath,sp1)# 子文件夹绝对路径
        sub1outpath=os.path.join(outPath,sp1)# 输出子文件/文件夹绝对路径

        if os.path.isfile(sub1inpath):# 是文件就复制
            shutil.copyfile(sub1inpath,sub1outpath) #移动用shutil.move（），下同
        else:# 是子文件夹就继续遍历
            if os.path.exists(sub1outpath)==False:#在输出路径下对应新建子文件夹
                os.mkdir(sub1outpath)

            for sp2 in os.listdir(sub1inpath):
                sub2Inpath=os.path.join(sub1inpath,sp2)# 二级子文件夹绝对路径
                sub2Outpath=os.path.join(sub1outpath,sp2)# 输出子文件/文件夹绝对路径

                if os.path.isfile(sub2Inpath):# 是文件就复制
                    shutil.copyfile(sub2Inpath,sub2Outpath) #移动用shutil.move（），下同
                else:# 是子文件夹就继续遍历，此处简单认为只有两层，直接跳出
                    continue
                    # if os.path.exists(sub2Outpath)==False:#在输出路径下对应新建子文件夹
                    #     os.mkdir(sub2Outpath)
    return 0

def main():
    return 0
main()