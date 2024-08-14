from ast import literal_eval
import pandas as pd
import sys, os

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('AAAI_expetiments'))))


df_pinn = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_PINN.csv')
df_PSO = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_PSO.csv')
df_lam = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_lam.csv')
df_F = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_fourier.csv')
df_natg = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_NGD.csv')
df_kan_PINN = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_PINN_KAN.csv')
df_kan_PSO = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_PSO_KAN.csv')
df_KAN_lam = pd.read_csv('examples\\AAAI_expetiments\\results\\burgers_lam_KAN.csv')

show_separate=False

if show_separate:

    plt.plot(df_pinn['grid_res'], df_pinn['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_pinn['grid_res'], df_pinn['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_pinn['grid_res'], df_pinn['lu_f_adam'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_pinn['grid_res'], df_pinn['loss_adam'], '-o', label='loss PINN')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV PINN')
    plt.legend()
    plt.show()






    plt.plot(df_PSO['grid_res'], df_PSO['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_PSO['grid_res'], df_PSO['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_PSO['grid_res'], df_PSO['lu_f_pso'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_PSO['grid_res'], df_PSO['loss_pso'], '-o', label='loss PSO')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV PINN PSO enchansment')
    plt.legend()
    plt.show()






    plt.plot(df_lam['grid_res'], df_lam['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_lam['grid_res'], df_lam['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_lam['grid_res'], df_lam['lu_f_adam'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_lam['grid_res'], df_lam['loss'], '-o', label='loss PINN')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV PINN adaptive lambdas')
    plt.legend()
    plt.show()





    plt.plot(df_F['grid_res'], df_F['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_F['grid_res'], df_F['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_F['grid_res'], df_F['lu_f_adam'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_F['grid_res'], df_F['loss'], '-o', label='loss PINN')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV PINN Fourier embedding')
    plt.legend()
    plt.show()







    plt.plot(df_natg['grid_res'], df_natg['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_natg['grid_res'], df_natg['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_natg['grid_res'], df_natg['lu_f_adam'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_natg['grid_res'], df_natg['loss'], '-o', label='loss PINN')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV NGD')
    plt.legend()
    plt.show()






    plt.plot(df_kan_PINN['grid_res'], df_kan_PINN['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_kan_PINN['grid_res'], df_kan_PINN['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_kan_PINN['grid_res'], df_kan_PINN['lu_f_adam'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_kan_PINN['grid_res'], df_kan_PINN['loss_adam'], '-o', label='loss PINN')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV PINN KAN')
    plt.legend()
    plt.show()






    plt.plot(df_kan_PSO['grid_res'], df_kan_PSO['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_kan_PSO['grid_res'], df_kan_PSO['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_kan_PSO['grid_res'], df_kan_PSO['lu_f_pso'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_kan_PSO['grid_res'], df_kan_PSO['loss_pso'], '-o', label='loss PSO')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV PINN PSO enchansment KAN')
    plt.legend()
    plt.show()



    plt.plot(df_KAN_lam['grid_res'], df_KAN_lam['error_train']**2, '-o', label='l2_error_train')
    plt.plot(df_KAN_lam['grid_res'], df_KAN_lam['error_test']**2, '-o', label='l2_error_test')
    plt.plot(df_KAN_lam['grid_res'], df_KAN_lam['lu_f_adam'], '-o', label='(lu-f,lu-f)')
    plt.plot(df_KAN_lam['grid_res'], df_KAN_lam['loss'], '-o', label='loss PINN')
    plt.yscale('log')
    #plt.xscale('log')
    plt.title('KDV PINN adaptive lambdas KAN')
    plt.legend()
    plt.show()


plt.figure(figsize=(20,10))
plt.plot(df_natg['grid_res'], df_natg['error_test']**2, '-o', label='l2_norm natural gradient')
plt.plot(df_lam['grid_res'], df_lam['error_test']**2, '-o', label='l2_norm adaptive lambdas')
plt.plot(df_PSO['grid_res'], df_PSO['error_test']**2, '-o', label='l2_norm pso enchansment')
plt.plot(df_pinn['grid_res'], df_pinn['error_test']**2, '-o', label='l2_norm classical PINN')
plt.plot(df_F['grid_res'], df_F['error_test']**2, '-o', label='l2_norm Fourier embedding')
#plt.plot(df_kan_PSO['grid_res'], df_kan_PSO['error_test']**2, '-o', label='l2_norm pso enchansment (KAN)')
#plt.plot(df_kan_PINN['grid_res'], df_kan_PINN['error_test']**2, '-o', label='l2_norm classical PINN (KAN)')
plt.plot(df_KAN_lam['grid_res'], df_KAN_lam['error_test']**2, '-o', label='l2_norm adaptive lambdas (KAN)')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of grid points')
plt.ylabel('l2 error')
plt.title('Burgers comparison test')
plt.legend()
plt.show()


##plt.plot()
##plt.plot(df_natg['grid_res'], df_natg['l2_error_train']**2, '-o', label='l2_norm natural gradient')
##plt.plot(df_lam['grid_res'], df_lam['l2_error_train']**2, '-o', label='l2_norm adaptive lambdas')
##plt.plot(df_PSO['grid_res'], df_PSO['l2_error_train']**2, '-o', label='l2_norm pso enchansment')
##plt.plot(df_pinn['grid_res'], df_pinn['l2_error_train']**2, '-o', label='l2_norm classical PINN')
##plt.plot(df_mat['grid_res'], df_mat['l2_error_train']**2, '-o', label='l2_norm_mat')
##plt.yscale('log')
##plt.xscale('log')
##plt.title('Poisson comparison train')
##plt.legend()
##plt.show()