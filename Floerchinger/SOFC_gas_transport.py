# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:26:41 2020

@author: gusfl
"""
import numpy as np



from scipy.integrate import solve_ivp #integration function for ODE system.
#from SOFC_function_1D import residual # point the model to the residual function
from SOFC_inputs import param, SV_0, ptr
   
# if i_ext:
#     param.i_ext = i_ext
# if P:
#     param.P_an = P

solution = solve_ivp(lambda t, y: residual(t, y, param, ptr),param.time_span, SV_0, rtol=1e-9, atol=1e-7, method='BDF')







def residual(t,SV,param,ptr):
    dSV_dt = np.zeros_like(SV)
        
    C_k_an_sub = SV[ptr.C_k_an_sub]
    C_k_an_CL = SV[ptr.C_k_an_CL]
    
    
    # State variables for node 1: substrate
    state1 = {'C_k':C_k_an_sub, 'T':param.T,'dY': param.th_sub,'mu': param.mu_g_an}
    # State variables for node 2: anode CL
    state2 = {'C_k':C_k_an_CL, 'T':param.T, 'dY': param.th_an, 'mu': param.mu_g_an}
    

    D_k_g_an = np.array([5.48e-4, 6.13e-5])
    
    # Gas properties
    gas_props = {'D_k':D_k_g_an, 'mu': param.mu_g_an}

    N_k_an = SOFC_gas_transport(state1,state2,gas_props)
    
    sdot_k_an = i_far_an*param.nu_k_an/param.n/F
    

    dSV_dt[ptr.C_k_an_CL] = (N_k_an + sdot_k_an*param.A_fac_an)/state2['dY']/param.eps_g_CL
    
    
    
    
    
def SOFC_gas_transport(state1, state2, gas_props):
    
    N_k = np.zeros_like(state1['C_k'])
    
    f1 = state1['dY']/(state1['dY'] + state2['dY'])
    f2 = 1-f1

    C_int = f1*state1['C_k'] + f2*state2['C_k']

    X_k_1 = state1['C_k']/np.sum(state1['C_k'])
    X_k_2 = state2['C_k']/np.sum(state2['C_k'])
    X_k_int = f1*X_k_1 + f2*X_k_2
    
    dY = (state1['dY'] + state2['dY'])/2
  
    # #Convection velocity
    V_conv_k = 0 

    # #Diffusion Velocity
    V_difn_k = -gas_props['D_k'] * (X_k_1-X_k_2)/dY/X_k_int

    #total flux
    N_k = C_int*X_k_int*(V_conv_k + V_difn_k)

    return N_k