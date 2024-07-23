from ast import literal_eval
import pandas as pd
import sys, os

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('AAAI_expetiments'))))

df_pinn = pd.read_csv('examples\\AAAI_expetiments\\results\\poisson_PINN.csv')
y = []

for i in df_pinn['loss']:
    y.append(literal_eval(i)[0])
# plt.plot(df['grid_res'], df_pinn['l2_norm']**2, '-*', label='l2_norm: PINN')
plt.plot(df_pinn['grid_res'], y, '-o', label='loss PINN')
plt.plot(df_pinn['grid_res'], df_pinn['l2_error_train']**2, '-o', label='l2_error_train')
plt.plot(df_pinn['grid_res'], df_pinn['l2_error_test']**2, '-o', label='l2_error_test')
plt.plot(df_pinn['grid_res'], df_pinn['lu_f'], '-o', label='(lu-f,lu-f)')
plt.plot(df_pinn['grid_res'], df_pinn['lu'], '-o', label='(lu,lu)=(f,f)')
plt.yscale('log')
plt.xscale('log')
plt.title('Poisson autograd PINN')
plt.legend()
plt.show()


df_PSO = pd.read_csv('examples\\AAAI_expetiments\\results\\poisson_PSO.csv')
y = []

for i in df_PSO['loss']:
    y.append(literal_eval(i)[0])
# plt.plot(df['grid_res'], df_pinn['l2_norm']**2, '-*', label='l2_norm: PINN')
plt.plot(df_PSO['grid_res'], y, '-o', label='loss PINN')
plt.plot(df_pinn['grid_res'], df_pinn['l2_error_train']**2, '-o', label='l2_error_train_PINN')
plt.plot(df_pinn['grid_res'], df_pinn['l2_error_test']**2, '-o', label='l2_error_test_PINN')
plt.plot(df_PSO['grid_res'], df_PSO['l2_error_train']**2, '-o', label='l2_error_train_PSO')
plt.plot(df_PSO['grid_res'], df_PSO['l2_error_test']**2, '-o', label='l2_error_test_PSO')
plt.plot(df_PSO['grid_res'], df_PSO['lu_f'], '-o', label='(lu-f,lu-f)')
plt.plot(df_PSO['grid_res'], df_PSO['lu'], '-o', label='(lu,lu)=(f,f)')
plt.yscale('log')
plt.xscale('log')
plt.title('Poisson autograd PSO')
plt.legend()
plt.show()


df_lam = pd.read_csv('examples\\AAAI_expetiments\\results\\poisson_lam.csv')
y = []

for i in df_lam['loss']:
    y.append(literal_eval(i)[0])
# plt.plot(df['grid_res'], df_pinn['l2_norm']**2, '-*', label='l2_norm: PINN')
plt.plot(df_lam['grid_res'], y, '-o', label='loss PINN')
plt.plot(df_lam['grid_res'], df_lam['l2_error_train']**2, '-o', label='l2_error_train_lam')
plt.plot(df_lam['grid_res'], df_lam['l2_error_test']**2, '-o', label='l2_error_test_lam')
plt.plot(df_lam['grid_res'], df_lam['lu_f'], '-o', label='(lu-f,lu-f)')
plt.plot(df_lam['grid_res'], df_lam['lu'], '-o', label='(lu,lu)=(f,f)')
plt.yscale('log')
plt.xscale('log')
plt.title('Poisson adaptive labmdas')
plt.legend()
plt.show()


df_mat = pd.read_csv('examples\\AAAI_expetiments\\results\\poisson_mat.csv')
y = []

for i in df_mat['loss']:
    y.append(literal_eval(i)[0])
# plt.plot(df['grid_res'], df_pinn['l2_norm']**2, '-*', label='l2_norm: PINN')
plt.plot(df_mat['grid_res'], y, '-o', label='loss mat')
plt.plot(df_mat['grid_res'], df_mat['l2_error_train']**2, '-o', label='l2_error_train_mat')
#plt.plot(df_natg['grid_res'], df_natg['lu_f'], '-o', label='(lu-f,lu-f)')
#plt.plot(df_natg['grid_res'], df_natg['lu'], '-o', label='(lu,lu)=(f,f)')
plt.yscale('log')
plt.xscale('log')
plt.title('Poisson grid point optimization')
plt.legend()
plt.show()


df_natg = pd.read_csv('examples\\AAAI_expetiments\\results\\poisson_NGD.csv')
y = []

#for i in df_natg['loss']:
#    y.append(literal_eval(i)[0])
## plt.plot(df['grid_res'], df_pinn['l2_norm']**2, '-*', label='l2_norm: PINN')
plt.plot(df_natg['grid_res'], df_natg['loss'], '-o', label='loss PINN')
plt.plot(df_natg['grid_res'], df_natg['l2_error_train']**2, '-o', label='l2_error_train')
plt.plot(df_natg['grid_res'], df_natg['l2_error_test']**2, '-o', label='l2_error_test')
plt.plot(df_natg['grid_res'], df_natg['lu_f'], '-o', label='(lu-f,lu-f)')
plt.plot(df_natg['grid_res'], df_natg['lu'], '-o', label='(lu,lu)=(f,f)')
plt.yscale('log')
plt.xscale('log')
plt.title('Poisson autograd NGD')
plt.legend()
plt.show()



plt.plot()
plt.plot(df_natg['grid_res'], df_natg['l2_error_test']**2, '-o', label='l2_norm natural gradient')
plt.plot(df_lam['grid_res'], df_lam['l2_error_test']**2, '-o', label='l2_norm adaptive lambdas')
plt.plot(df_PSO['grid_res'], df_PSO['l2_error_test']**2, '-o', label='l2_norm pso enchansment')
plt.plot(df_pinn['grid_res'], df_pinn['l2_error_test']**2, '-o', label='l2_norm classical PINN')
plt.plot(df_mat['grid_res'], df_mat['l2_error_test']**2, '-o', label='l2_norm_mat')
plt.yscale('log')
plt.xscale('log')
plt.title('Poisson comparison test')
plt.legend()
plt.show()


plt.plot()
plt.plot(df_natg['grid_res'], df_natg['l2_error_train']**2, '-o', label='l2_norm natural gradient')
plt.plot(df_lam['grid_res'], df_lam['l2_error_train']**2, '-o', label='l2_norm adaptive lambdas')
plt.plot(df_PSO['grid_res'], df_PSO['l2_error_train']**2, '-o', label='l2_norm pso enchansment')
plt.plot(df_pinn['grid_res'], df_pinn['l2_error_train']**2, '-o', label='l2_norm classical PINN')
plt.plot(df_mat['grid_res'], df_mat['l2_error_train']**2, '-o', label='l2_norm_mat')
plt.yscale('log')
plt.xscale('log')
plt.title('Poisson comparison train')
plt.legend()
plt.show()