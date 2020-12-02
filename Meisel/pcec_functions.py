#This is where I will store my functions to be used in my model


#/\/\/\/\/\ Imports: /\/\/\/\/\
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
import math

#Constants:
F = 96485 #C/mol
R = 8.313 #J/mol*k

def residual(t, SV, pars, ptr):
    dSV_dt = np.empty_like(SV) #initializing residual (Zeroes_like gave me errors)
    
    #----- Negatrode -----
    dphi_neg = SV[ptr.dphi_int_neg_0]
    
    #Mass Action Equations:
    i_Far_neg_o = pars.n_neg_o*F*(pars.k_fwd_star_neg_o*math.exp((-pars.beta_o*pars.n_neg_o*F*dphi_neg)/(R*pars.T))*pars.prod_fwd_neg_o-pars.k_rev_star_neg_o*math.exp(((1-pars.beta_o)*pars.n_neg_o*F*dphi_neg)/(R*pars.T)))
    i_Far_neg_p = pars.n_neg_p*F*(pars.k_fwd_star_neg_p*math.exp((-pars.beta_p*pars.n_neg_p*F*dphi_neg)/(R*pars.T))*pars.prod_fwd_neg_p-pars.k_rev_star_neg_p*math.exp(((1-pars.beta_p)*pars.n_neg_p*F*dphi_neg)/(R*pars.T)))
    i_Far_neg = i_Far_neg_o + i_Far_neg_p
    
    #Final calculations for the change in potential difference on the negatrode
    i_dl_neg = pars.i_ext - i_Far_neg
    ddphi_int_neg = -i_dl_neg/pars.C_dl_neg
    dSV_dt[ptr.dphi_int_neg_0] = ddphi_int_neg

    #----- Positrode -----
    dphi_pos = SV[1]
    
    #Mass-Action equations
    i_Far_pos_o = pars.n_pos_o*F*(pars.k_fwd_star_pos_o*math.exp((-pars.beta_o*pars.n_pos_o*F*dphi_pos)/(R*pars.T))*pars.prod_fwd_pos_o-pars.k_rev_star_pos_o*math.exp(((1-pars.beta_o)*pars.n_pos_o*F*dphi_pos)/(R*pars.T)))
    i_Far_pos_p = pars.n_pos_p*F*(pars.k_fwd_star_pos_p*math.exp((-pars.beta_p*pars.n_pos_p*F*dphi_pos)/(R*pars.T))*pars.prod_fwd_pos_p-pars.k_rev_star_pos_p*math.exp(((1-pars.beta_p)*pars.n_pos_p*F*dphi_pos)/(R*pars.T)))
    i_Far_pos = i_Far_pos_o + i_Far_pos_p
    
    i_dl_pos = pars.i_ext - i_Far_pos
    ddphi_int_pos = -i_dl_pos/pars.C_dl_pos
    dSV_dt[1] = ddphi_int_pos
    return dSV_dt