# coding:gbk
import os,shutil
def fileCopy():#
    inPath=""
    outPath=""

    for sp1 in os.listdir(inPath):# �������ļ��У�sp���ļ�����
        sub1inpath=os.path.join(inPath,sp1)# ���ļ��о���·��
        sub1outpath=os.path.join(outPath,sp1)# ������ļ�/�ļ��о���·��

        if os.path.isfile(sub1inpath):# ���ļ��͸���
            shutil.copyfile(sub1inpath,sub1outpath) #�ƶ���shutil.move��������ͬ
        else:# �����ļ��оͼ�������
            if os.path.exists(sub1outpath)==False:#�����·���¶�Ӧ�½����ļ���
                os.mkdir(sub1outpath)

            for sp2 in os.listdir(sub1inpath):
                sub2Inpath=os.path.join(sub1inpath,sp2)# �������ļ��о���·��
                sub2Outpath=os.path.join(sub1outpath,sp2)# ������ļ�/�ļ��о���·��

                if os.path.isfile(sub2Inpath):# ���ļ��͸���
                    shutil.copyfile(sub2Inpath,sub2Outpath) #�ƶ���shutil.move��������ͬ
                else:# �����ļ��оͼ����������˴�����Ϊֻ�����㣬ֱ������
                    continue
                    # if os.path.exists(sub2Outpath)==False:#�����·���¶�Ӧ�½����ļ���
                    #     os.mkdir(sub2Outpath)
    return 0

def main():
    return 0
main()