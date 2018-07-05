import numpy as np
import math
import random
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import loggamma
from scipy.stats import gamma
from scipy.stats import weibull_min
from scipy.stats import lognorm
from scipy.stats import expon
from scipy.stats import exponnorm
from scipy.stats import kstest
from scipy.stats import chisquare
from scipy.stats import alpha
from scipy.stats import chi2
from scipy.stats import pearson3
from scipy.stats import powerlaw
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

#TRMM_Data=np.loadtxt('C:/Users/xia/Desktop/新实验/200702/20070228_trmm_hb.txt')
#TRMM_Data=np.loadtxt('C:/Users/xia/Desktop/20090228CMORPHRH/20090228_logorm_no_zero_hb_GDA/200902283232_Gmf_ave5.txt')
#TRMM_Data=np.loadtxt('C:/Users/xia/Desktop/20090228CMORPHRH/200902283232_gauss01_no_zero_hb_mf.txt')
Downscaling_Data=np.loadtxt('C:/Users/xia/Desktop/新实验/统计分析/20100703_gauss_no_zero_hb_mf_ave.txt')
Raingauge_Data=np.loadtxt('C:/Users/xia/Desktop/新实验/统计分析/20100703_RG.txt')


Downscaling_Data_Average=sum(Downscaling_Data)/len(Downscaling_Data)
Raingauge_Data_Average=sum(Raingauge_Data)/len(Raingauge_Data)
sum1=0
sum2=0
sum3=0
sum4=0
sum5=0
sum6=0
sum7=0
for n in range(0,75):
        sum1=sum1+(Downscaling_Data[n]-Downscaling_Data_Average)*(Raingauge_Data[n]-Raingauge_Data_Average)
        sum2=sum2+(Downscaling_Data[n]-Downscaling_Data_Average)**2
        sum3=sum3+(Raingauge_Data[n]-Raingauge_Data_Average)**2
        sum4=sum4+(Raingauge_Data[n]-Downscaling_Data[n])**2
        sum5=sum5+abs(Raingauge_Data[n]-Downscaling_Data[n])
        sum6=sum6+Raingauge_Data[n]
        sum7=sum7+Downscaling_Data[n]

R=sum1/math.sqrt(sum2*sum3)
RMSE=math.sqrt(sum4/75)
Bias=sum6/sum7-1
MAE=sum5/75
print("R:",R)
print("RMSE:",RMSE)
print("Bias:",Bias)
print("MAE:",MAE)


TRMM_no_zero=filter(lambda x:x!=0,TRMM_Data)
TRMM_no_zero_list=list(TRMM_no_zero)
#TRMM_no_zero_list_sum=np.sum(TRMM_no_zero_list)
#if (TRMM_no_zero_list_sum > 0):
#        TRMM_no_zero_list_normlized = TRMM_no_zero_list /TRMM_no_zero_list_sum
#else:
#        TRMM_no_zero_list_normlized= TRMM_no_zero_list
#TRMM_no_zero_list_inv=np.ones((len(TRMM_no_zero_list),1),np.float)
#for i in range(0,len(TRMM_no_zero_list)):
#        TRMM_no_zero_list_inv[i] = TRMM_no_zero_list_inv[i]/TRMM_no_zero_list[i]
#print(TRMM_no_zero_list)
#np.savetxt('C:/Users/xia/Desktop/新实验/200803/'+'20080320_trmm_no_zero.txt',TRMM_no_zero_list,fmt = '%.8f')

#n=10
#p=0.3
#binomial=binom.pmf(Raingauge_Data,n,p)
#poisson_test=poisson.pmf(Raingauge_Data,2)
norm_test=norm.fit(TRMM_no_zero_list)
#chisquare_norm=chisquare(norm_test)
#print("chisquare_norm：",chisquare_norm)
#beta_test=beta.fit(TRMM_no_zero_list)
#gengamma_test=gamma.fit(TRMM_no_zero_list)
#TRMM_Data_gamma=gengamma.cdf(TRMM_no_zero_list_normlized,gengamma_test[0],gengamma_test[1])
#plt.plot(TRMM_Data,TRMM_Data_gamma)
#weibull_min_test=weibull_min.fit(TRMM_Data)
#lognorm_test=lognorm.fit(TRMM_no_zero_list)
#expon_test=expon.fit(TRMM_Data)
#TRMM_Data_expon=expon.cdf(TRMM_Data,expon_test[0])
#chisquare_expon=chisquare(TRMM_Data_expon)
#kstest_gamma=kstest(TRMM_no_zero_list,'gamma',args=(gengamma_test[0],gengamma_test[1],gengamma_test[2]),N=len(TRMM_no_zero_list))
#kstest_inv_lognormal=kstest(TRMM_no_zero_list,'lognorm',args=(lognorm_test[0],lognorm_test[1],lognorm_test[2]),N=len(TRMM_no_zero_list))


#alpha = gengamma_test[0]
#beta = gengamma_test[2]
#aa_alpha = math.sqrt(alpha)
#bb_beta = math.sqrt(beta)
#e_aa_mu = np.exp(aa_mu)
#sqrt_bb_sigma = np.sqrt(bb_sigma)
#print(gengamma_test[0],gengamma_test[1],gengamma_test[2])
#print('alpha,beta:',alpha,beta)
#print("aa_alpha,bb_beta:",aa_alpha,bb_beta)

#print(kstest_gamma)
#print(kstest_norm)
#print("kstest_inv_lognormal:",kstest_inv_lognormal)
#exponnorm_test=exponnorm.fit(TRMM_Data)
#vals = gengamma.ppf([0.001, 0.5, 0.999],0.74,1.05)
#kstest_gamma=kstest(norm_test,'norm')
#chisquare_gamma=chisquare(TRMM_Data_gamma)
#print("chisquare_gamma:",TRMM_Data_gamma)
#plt.plot(TRMM_Data,gengamma.pdf(TRMM_Data,0.3,1.06),'r-',lw=5,alpha=0.6,label='gengamma pdf')
#plt.figure(0)
#plt.plot(TRMM_no_zero_list,TRMM_Data_gamma)
#plt.show()
#plt.figure(1)
#plt.hist(Raingauge_Data)
#plt.figure(2)
#plt.hist(Downscaling_Data)
#plt.figure(3)

#plt.hist(aaa)

#plt.show()

