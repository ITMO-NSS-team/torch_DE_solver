# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:55:32 2022

@author: user
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



sns.set(rc={'figure.figsize':(11.7,8.27)},font_scale=4)
sns.set_style("whitegrid")

df1=pd.read_csv('benchmarking_data/legendre_poly_exp_cache=False.csv',index_col=0)
df2=pd.read_csv('benchmarking_data/legendre_poly_exp_cache=True.csv',index_col=0)
df_L=pd.concat((df1,df2))


# plt.figure()
# sns.boxplot(x='type', y='RMSE', data=df_L, showfliers=False, hue='cache')


# plt.figure()
# sns.boxplot(x='type', y='time', data=df_L, showfliers=False, hue='cache')



fig, axes = plt.subplots(1, 2, figsize=(40,20))

sns.boxplot(ax=axes[0],x='type', y='time', data=df_L, showfliers=False, hue='cache')

sns.boxplot(ax=axes[1],x='type', y='RMSE', data=df_L, showfliers=False, hue='cache')



plt.savefig('Legendre_exps_NN.eps', format='eps',bbox_inches='tight')



df_L_mat=pd.read_csv('benchmarking_data/legendre_poly_exp_martix.csv',index_col=0)
df_L_mat['method']='matrix'
df2['method']='NN'
df_mat_compare=pd.concat((df2,df_L_mat))


# plt.figure()
# sns.boxplot(x='type', y='time', data=df_mat_compare, showfliers=False, hue='method')
# plt.figure()
# sns.boxplot(x='type', y='RMSE', data=df_mat_compare, showfliers=False, hue='method')

fig, axes = plt.subplots(1, 2, figsize=(40,20))

sns.boxplot(ax=axes[0],x='type', y='time', data=df_mat_compare, showfliers=False, hue='method')

sns.boxplot(ax=axes[1],x='type', y='RMSE', data=df_mat_compare, showfliers=False, hue='method')

plt.savefig('Legendre_exps_NN_vs_mat.eps', format='eps',bbox_inches='tight')



fig, axes = plt.subplots(2, 3, figsize=(60,30))

df1=pd.read_csv('benchmarking_data/PI_experiment_10_500_cache=False.csv',index_col=0)
df2=pd.read_csv('benchmarking_data/PI_experiment_10_500_cache=True.csv',index_col=0)
dfPI=pd.concat((df1,df2))

sns.boxplot(ax=axes[0,0],x='grid_res', y='RMSE', data=dfPI, showfliers=False, hue='cache')
axes[0,0].set_title('PI Error')
sns.boxplot(ax=axes[1,0],x='grid_res', y='time', data=dfPI, showfliers=False, hue='cache')
axes[1,0].set_title('PI Time')





df1=pd.read_csv('benchmarking_data/PII_experiment_10_500_cache=False.csv',index_col=0)
df2=pd.read_csv('benchmarking_data/PII_experiment_10_500_cache=True.csv',index_col=0)
dfPII=pd.concat((df1,df2))

sns.boxplot(ax=axes[0,1],x='grid_res', y='RMSE', data=dfPII, showfliers=False, hue='cache')
axes[0,1].set_title('PII Error')

sns.boxplot(ax=axes[1,1],x='grid_res', y='time', data=dfPII, showfliers=False, hue='cache')
axes[1,1].set_title('PII Time')




df1=pd.read_csv('benchmarking_data/PIII_experiment_100_500_cache=False.csv',index_col=0)
df2=pd.read_csv('benchmarking_data/PIII_experiment_100_500_cache=True.csv',index_col=0)
dfPIII=pd.concat((df1,df2))


sns.boxplot(ax=axes[0,2],x='grid_res', y='RMSE', data=dfPIII, showfliers=False, hue='cache')
axes[0,2].set_title('PIII Error')

sns.boxplot(ax=axes[1,2],x='grid_res', y='time', data=dfPIII, showfliers=False, hue='cache')
axes[1,2].set_title('PIII Time')

plt.savefig('PI-PIII_exps.eps', format='eps',bbox_inches='tight')


dfPIV=pd.read_csv('benchmarking_data/PIV_experiment_100_500_cache=True.csv',index_col=0)

# plt.figure()

# sns.scatterplot(x='grid_res', y='RMSE', data=dfPIV, hue='cache')

# plt.figure()

# sns.scatterplot(x='grid_res', y='time', data=dfPIV, hue='cache')


dfPV=pd.read_csv('benchmarking_data/PV_experiment_100_500_cache=True.csv',index_col=0)

# plt.figure()

# sns.scatterplot(x='grid_res', y='RMSE', data=dfPV, hue='cache')

# plt.figure()

# sns.scatterplot(x='grid_res', y='time', data=dfPV, hue='cache')


dfPVI=pd.read_csv('benchmarking_data/PVI_experiment_100_500_cache=True.csv',index_col=0)

# plt.figure()

# sns.scatterplot(x='grid_res', y='RMSE', data=dfPVI, hue='cache')

# plt.figure()

# sns.scatterplot(x='grid_res', y='time', data=dfPVI, hue='cache')


dfPanleve=pd.concat((dfPI,dfPII,dfPIII,dfPIV,dfPV,dfPVI))
dfPanleve=dfPanleve.drop(columns='index')
dfPanleve=dfPanleve.reset_index(drop=True)



compare_list=[]

for grid_res in range(100,501,100):
    for eq_type in ['PI','PII','PIII','PIV','PV','PVI']:
        mean_RMSE=np.mean(dfPanleve[(dfPanleve['grid_res']==grid_res) & (dfPanleve['type']==eq_type)]['RMSE']/np.max(dfPanleve[(dfPanleve['grid_res']>=100) & (dfPanleve['type']==eq_type)]['RMSE']))
        mean_time=np.log10(np.mean(dfPanleve[(dfPanleve['grid_res']==grid_res) & (dfPanleve['type']==eq_type)]['time']))
        compare_list.append({'type':eq_type,'grid_res':grid_res,'mean_RMSE':mean_RMSE,'mean_time':mean_time})

df_compare=pd.DataFrame(compare_list)

# plt.figure()

# ax=sns.lineplot(x='grid_res', y='mean_RMSE', data=df_compare, hue='type')
# ax.set(xlabel='grid_res', ylabel='mean(RMSE)')

# plt.figure()

# ax=sns.lineplot(x='grid_res', y='mean_time', data=df_compare, hue='type')
# ax.set(xlabel='grid_res', ylabel='log(time)')
# plt.show()


fig, axes = plt.subplots(1, 2, figsize=(40,20))


sns.lineplot(ax=axes[0],x='grid_res', y='mean_time', data=df_compare, hue='type')
axes[0].set(xlabel='grid_res', ylabel='log(time)')


sns.lineplot(ax=axes[1],x='grid_res', y='mean_RMSE', data=df_compare, hue='type')
axes[1].set(xlabel='grid_res', ylabel='mean(RMSE)')



plt.savefig('Panleve_exps_full_mean.eps', format='eps',bbox_inches='tight')

fig, axes = plt.subplots(2, 3, figsize=(60,30))

wave_df=pd.read_csv('benchmarking_data/wave_experiment_2_20_100_cache=True.csv',index_col=0)




sns.boxplot(ax=axes[0,0],x='grid_res', y='RMSE', data=wave_df, showfliers=False,hue='cache')
axes[0,0].set_title('WE Error')
sns.boxplot(ax=axes[1,0],x='grid_res', y='time', data=wave_df, showfliers=False,hue='cache')
axes[1,0].set_title('WE Time')

heat_df=pd.read_csv('benchmarking_data/heat_experiment_10_100_cache=True.csv',index_col=0)

sns.boxplot(ax=axes[0,1],x='grid_res', y='RMSE', data=heat_df, showfliers=False,hue='cache')
axes[0,1].set_title('HE Error')
sns.boxplot(ax=axes[1,1],x='grid_res', y='time', data=heat_df, showfliers=False,hue='cache')
axes[1,1].set_title('HE Time')


kdv_df=pd.read_csv('benchmarking_data/kdv_experiment_10_30_cache=True.csv',index_col=0)

# kdv_df['cache']=True

sns.boxplot(ax=axes[0,2],x='grid_res', y='RMSE', data=kdv_df, showfliers=False,hue='cache')
axes[0,2].set_title('KdV Error')
sns.boxplot(ax=axes[1,2],x='grid_res', y='time', data=kdv_df, showfliers=False,hue='cache')
axes[1,2].set_title('KdV Time')

plt.savefig('PDE_exps.eps', format='eps',bbox_inches='tight')



df_kdv_solitary=pd.read_csv('benchmarking_data/kdv_solitary_experiment_30_100_cache=True.csv',index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(40,20))

sns.boxplot(ax=axes[0],x='grid_res', y='time', data=df_kdv_solitary, showfliers=False, hue='type')

sns.boxplot(ax=axes[1],x='grid_res', y='RMSE', data=df_kdv_solitary, showfliers=False, hue='type')

plt.savefig('kdv_solitary.eps', format='eps',bbox_inches='tight')


df_kdv_solitary=pd.read_csv('benchmarking_data/wave_experiment_physical_10_100_cache=True.csv',index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(40,20))

sns.boxplot(ax=axes[0],x='grid_res', y='time', data=df_kdv_solitary, showfliers=False, hue='type')

sns.boxplot(ax=axes[1],x='grid_res', y='RMSE', data=df_kdv_solitary, showfliers=False, hue='type')

plt.savefig('wave_physics.eps', format='eps',bbox_inches='tight')