# -*- coding: utf-8 -*-
"""
This function creates a resirual vector for  1-D IT-SOFC cell. The function 
uses Butler Volmer kinetics to resolve current densities and (eventually ) 
incorporates the Dusty gas model to model massdiffusion of species.


@author: gusfl
"""
import numpy as np
from math import exp


####constants and useful values######

R= 8.3145 #universal gas constant [J/mol-K]
F = 96485 #Faradays constant [C/mol]


#Solution vector takes the form: [phi_an_DL,phi_ca_DL,C_k_an,C_k_ca]

def residual (t,SV,param,ptr):

    dSV_dt = np.zeros_like(SV)
    
    P_H2 = SV[ptr.C_k_an_CL][1]*R*param.T/100000
    #print(P_H2)
    ###########charge transport ###################3
    "Anode"    
    
    
    #Activation Overpotental
    eta_an = SV[ptr.phi_dl_an] - param.delta_phi_an_eq
    

    #The equation for anode exchange current density is given in Leah et al
    i_o_an = (P_H2)**0.5 * param.K_a* (R*param.T)/(param.n*F)*exp(-param.E_a_an/(R*param.T))

    #Butler-Volmer 
    i_far_an = i_o_an*(exp((param.alpha_an_a*param.n*F*eta_an)/(R*param.T)) - 
                       exp((-param.alpha_an_c*param.n*F*eta_an)/(R*param.T)))

    #print(eta_an)
    i_dl_an = i_far_an + param.i_ext
    #governing eqns
    dSV_dt[ptr.phi_dl_an] = -i_dl_an/param.C_dl_an 
    
    "Cathode"    
    #Activation Overpotental
    eta_ca = SV[ptr.phi_dl_ca] - param.delta_phi_ca_eq
    #The equation for anode exchange current density is given in Leah et al
    i_o_ca = param.K_c* (R*param.T)/(param.n*F)*exp(-param.E_a_ca/(R*param.T))
    #Butler-Volmer 
    i_far_ca = i_o_ca*(exp((param.alpha_ca_a*param.n*F*eta_ca)/(R*param.T)) - 
                       exp((-param.alpha_ca_c*param.n*F*eta_ca)/(R*param.T)))


    i_dl_ca = i_far_ca + param.i_ext

    #governing eqns
    dSV_dt[ptr.phi_dl_ca] = -i_dl_ca/param.C_dl_ca 
    
   ############Species Flux#################### 
    
    C_k_an_sub = SV[ptr.C_k_an_sub]
    
    C_k_an_CL = SV[ptr.C_k_an_CL]
    
    
    # State variables for node 1: substrate
    state1 = {'C_k':C_k_an_sub, 'T':param.T,'dY': param.th_sub,'mu': param.mu_g_an}
    # State variables for node 2: anode CL
    state2 = {'C_k':C_k_an_CL, 'T':param.T, 'dY': param.th_an, 'mu': param.mu_g_an}
    
    
    D_k_g_an = np.array([5.48e-4, 6.13e-5])
    
    # Gas properties
    gas_props = {'D_k':D_k_g_an, 'mu': param.mu_g_an}

    N_k_an = SOFC_gas_transport(state1,state2,gas_props,param)
    
    sdot_k_an = i_far_an*param.nu_k_an/param.n/F
    #print(N_k_an)

    dSV_dt[ptr.C_k_an_CL] = (N_k_an + sdot_k_an*param.A_fac_an)/state2['dY']/param.eps_g_CL


    return dSV_dt


def SOFC_gas_transport(state1, state2, gas_props, param):
    
    from Diffusion_function import  anode_difn_coeffs
    
    N_k = np.zeros_like(state1['C_k'])
    
    f1 = state1['dY']/(state1['dY'] + state2['dY'])
    f2 = 1-f1

    C_int = f1*state1['C_k'] + f2*state2['C_k']

    #print(state1['C_k'])
    #print(state2['C_k'])

    X_k_1 = state1['C_k']/np.sum(state1['C_k'])
    #print(X_k_1)
    
    X_k_2 = state2['C_k']/np.sum(state2['C_k'])
    #print(X_k_2)
    
    
    X_k_int = f1*X_k_1 + f2*X_k_2
    
    P_1 = np.sum(state1['C_k'])*R*state1['T']
    P_2 = np.sum(state2['C_k'])*R*state2['T']
    
    dY = (state1['dY'] + state2['dY'])/2
  
    D_k_g_an_calc = anode_difn_coeffs(X_k_2, param)
    
    D_k_g_an_calc = np.reshape(D_k_g_an_calc,(1,2))
    
    #print(D_k_g_an_calc)
    # #Convection velocity
    V_conv_k = 0
    
    
    # #Diffusion Velocity
    V_difn_k = -D_k_g_an_calc * (X_k_2-X_k_1)/dY/X_k_int
    
    #print(V_difn_k)
    #total flux
    N_k = C_int*X_k_int*(V_conv_k + V_difn_k)

    return N_k







                    