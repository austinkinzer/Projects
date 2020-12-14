# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:35:30 2020

@author: gfloerchinger
"""

import numpy as np

# T = 873
# P = 101325
R = 8.3145




"anode"


def anode_difn_coeffs(X_f, param):
    
    #def difn_coeff(param,gas_props):
    N = len(X_f)
        
    D_m_ij = np.zeros((N,N))
    D_kn_i = np.zeros((N,1))
    D_m_k_an = np.zeros((N,1))
    
    
    for i, k in enumerate(D_m_ij):
        #knudsen Diffusion
    
        D_kn_i[i] = 2/3*param.d_pore_an * np.sqrt(8*R*param.T/np.pi/param.MM_f[i])
        #print(D_kn_i)
        for j, l in enumerate(k):
             #Fuller Approximation for effective mixture diffusion   
            #print(i,j)
            MM_ij = 2*(1/param.MM_f[i] + 1/param.MM_f[j])**(-1)
            D_m_ij[i,j] = 0.00143*param.T**(1.75) / (param.P_an*MM_ij**0.5 * (param.V_f[i]**(1/3) + (param.V_f[j]**(1/3))))
            
         
        #mixture averaged diffusion coeff    
        D_m_k_an[i] = np.sum(x for l,x in enumerate(np.divide(param.X_f,D_m_ij[i,:])) if l!=i)
        D_m_k_an[i] = (1-param.X_f[i])/D_m_k_an[i]
    
    
    #Effective diffusion coefficent takes knudsen and mixture difn in series
    D_eff_k_an = param.eps_an/param.tau_an * 1./(1./D_kn_i + 1./D_m_k_an)
        
    return D_eff_k_an




