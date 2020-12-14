# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:27:29 2020

@author: Lydia Meyer
"""
#anode only
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
from math import exp, log
from sofc_model_dec9 import solution


C_lat = 22325 #mol/m3
X_H_Ni = 0.6
X_H2O_Ni = 0.2
X_Vac_Ni = 0.2
X_Vac_elyte = 0.08
X_Ox_elyte = 0.92
# Species order:  vacancy, hydrogen, water
X_k_1 = np.array([0.2, 0.6, 0.2])
X_k_2 = np.array([0.018, 0.938, 0.34]) #complete guesses

z_k = np.array([-2., 1., 0.])

T = 298  #K

dY = 20e-6

D_k = np.array([1.28e-12, 0., 7.46e-11])

R = 8.3145 # J/mol-K
F = 96485  # C/mol equiv

sigma_elyte=0.001
phi_1 = 0
phi_2 = 0.6
P_1 = 101325 # Pa
P_2 = 100000 # Pa

# # State variables for node 1:
state1 = {'X_k':X_k_1, 'phi':phi_1, 'T':T}
state2 = {'X_k':X_k_2, 'phi':phi_2, 'T':T } 

geom = {'dY':dY }
ceramic_pars = { 'D_k':D_k,'z_k':z_k,'C_lat':C_lat,'sigma_elyte':sigma_elyte}



def sofc_transport(state1, state2, geom, ceramic_pars):
    c_k_1=state1['X_k']*ceramic_pars['C_lat']
    c_k_2=state2['X_k']*ceramic_pars['C_lat']
    c_k_avg=(c_k_1+c_k_2)/2
    N_diff=(-ceramic_pars['D_k']*(c_k_2-c_k_1))/geom['dY']
    N_mig=-ceramic_pars['z_k']*ceramic_pars['D_k']/R/T*F*c_k_avg*(state2['phi']-state1['phi'])/geom['dY']
    N_k=N_diff+N_mig
    

    
    return N_k
dPhi = solution.y
eta_Far = np.zeros_like(dPhi)
i_tot = np.zeros_like(dPhi)

# for ii in dPhi:
#     print(ii)

# for ii in dPhi:
test_phi=np.array(dPhi[0,1])
Nk_array=[]
for ii in dPhi:
    state2_array = np.array(state1['phi']-ii)


for ii in state2_array:
    state2['phi']=ii
    Nk= sofc_transport(state1, state2, geom, ceramic_pars)
    Nk_array.append(Nk)
    # f3=np.transpose(Nk)
    # plt.plot(dPhi,f3)
    # i_ion = np.dot(Nk,ceramic_pars['z_k'])*F
    
    # eta_Far[j] = 100*i_ion/i_tot[j]
    
    N_k_vec=np.stack( Nk_array, axis=0 )   
#Plot the results:
plt.plot(np.transpose(dPhi),Nk_array)
plt.xlabel('Potential (V)',fontsize=14)
plt.ylabel('Flux (mol/m^2/s)',fontsize=14)
plt.title('Anode transport at various potentials')
plt.legend(['Nickel vacancy','hydrogen ions','water in Ni'])
plt.show()



#GDL
D_k_gas = np.array([2.438e-5, 2.798e-5, 1.9e-5]) #m2/s
mu = 2.08e-5 #kg/m-s
gas_props = {'D_k':D_k_gas, 'mu':mu}
dY = 100e-6 # m
eps_g = 0.57
n_Brugg = -0.5

d_part = 0.5e-6
r_p = 2e-6
geom = {'eps_g':eps_g, 'n_Brugg':n_Brugg, 'd_part':d_part, 'dY':dY}
# State variables for node 1:
state1 = {'X_k':X_k_1, 'P':P_1, 'T':T}
# State variables for node 2:
state2 = {'X_k':X_k_2, 'P':P_2, 'T':T}


def sofc_gas_transport(state1, state2, geom, gas_props):
    N_k = np.zeros_like(state1['X_k'])
    tau_fac=geom['eps_g']**geom['n_Brugg']
    
    D_k_eff=gas_props['D_k']*geom['eps_g']/tau_fac
    k_g=(geom['eps_g']**3)*geom['d_part']**2/(72*(tau_fac**2)*(1-geom['eps_g'])**2)
    c_k_2=state2['X_k']*state2['P']/R/state2['T']
    c_k_1=state1['X_k']*state1['P']/R/state1['T']
    c_k_avg=(sum(c_k_1)+sum(c_k_2))/2
    x_k_avg=(state1['X_k']+state2['X_k'])/2
    
    v_conv=-k_g*(state2['P']-state1['P'])/gas_props['mu']/geom['dY']
    v_diff=-D_k_eff*(state2['X_k']-state1['X_k'])/geom['dY']/x_k_avg
    N_k=(v_conv+v_diff)*c_k_avg*x_k_avg
    
    
    
    return N_k
temps =np.linspace(300,1200,num=100)
N_k_gdl=[]
#vary temps
for T in temps:
    # State variables for node 1:
    state1 = {'X_k':X_k_1, 'P':P_1, 'T':T}
    # State variables for node 2:
    state2 = {'X_k':X_k_2, 'P':P_2, 'T':T}
    N_k_calc = sofc_gas_transport(state1, state2, geom, gas_props)
    N_k_gdl.append(N_k_calc)
    N_k_vec_gdl=np.stack( N_k_gdl, axis=0 )   

#Plot the results:
plt.plot(temps,N_k_vec_gdl)
plt.xlabel('Temperature (K)',fontsize=14)
plt.ylabel('Flux (mol/m^2/s)',fontsize=14)
plt.title('Anode transport at various temperatures')
plt.legend(['O2','H2','H2O'])
plt.show()


T=298
pressures =np.linspace(101325,303975,num=100)
N_k_presh=[]
#vary temps
for P in pressures:
    # State variables for node 1:
    state1 = {'X_k':X_k_1, 'P':P, 'T':T}
    # State variables for node 2:
    N_k_calc_p = sofc_gas_transport(state1, state2, geom, gas_props)
    N_k_presh.append(N_k_calc_p)
    N_k_vec_pressure=np.stack( N_k_presh, axis=0 )   

#Plot the results:
plt.plot(pressures,N_k_vec_pressure)
plt.xlabel('Pressure (Pa)',fontsize=14)
plt.ylabel('Flux (mol/m^2/s)',fontsize=14)
plt.title('Anode transport at various pressures')
plt.legend(['O2','H2','H2O'])
plt.show()
