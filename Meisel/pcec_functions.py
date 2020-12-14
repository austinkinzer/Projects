#This is where I will store my functions to be used in my model


#/\/\/\/\/\ Imports: /\/\/\/\/\
import numpy as np
import math

#Constants:
F = 96485 #C/mol
R = 8.313 #J/mol*k


#/\/\/\/\/\/\ Gas transport functions /\/\/\/\/\/\/\
#So far only contains one function (likely will stay that way)
def electrode_gas_transport(s1,s2,gasProps): #calculates gas diffusion betwee two nodes
    N_k  = np.zeros_like(s1['C_k'])
    #Setting the volume fractions of each layer of the negatrode:
    f1 = s1['dy']/(s1['dy'] + s2['dy'])
    f2 = 1-f1
    C_int = f1*s1['C_k'] + f2*s2['C_k']
    #re-finding the mol fractions of the gas constituents
    X_k_1 = s1['C_k']/np.sum(s1['C_k'])
    X_k_2 = s2['C_k']/np.sum(s2['C_k'])
    X_k_int = f1*X_k_1 + f2*X_k_2
    #Calculating the pressure values
    P_1 = np.sum(s1['C_k'])*R*gasProps['T']
    P_2 = np.sum(s2['C_k'])*R*gasProps['T']
    #Calculating V_k_diff
    D_k_eff = gasProps['eps_g_neg']*gasProps['D_k']/gasProps['tau_fac_neg'] #eps_g_neg and tau_fac_neg will be solved for before the function
    dY = 0.5*(s1['dy'] + s2['dy']) #getting the average thickness for each layer
    V_k_diff = -D_k_eff*(X_k_2 - X_k_1)/(dY*X_k_int)
    #Calculating Vconv and V_k)tot
    V_conv = -gasProps['Kg_neg']*(P_2 - P_1)/dY/gasProps['mu'] #Kg_neg will be solved for before the function
    V_k_diff = -D_k_eff*(X_k_2 - X_k_1)/dY/X_k_int

    V_k  = V_conv + V_k_diff
        
    N_k = C_int*X_k_int*V_k
    return N_k


#/\/\/\/\/\/\ Main Modeling function /\/\/\/\/\/\/\
def residual(t, SV, pars, ptr):
    dSV_dt = np.empty_like(SV) #initializing residual (Zeroes_like gave me errors)
    
    #----- Negatrode ------------------------------------------------
    "Charge Transfer"
    dphi_neg = SV[ptr.dphi_int_neg]
    #Mass Action Equations For Charge Transfer:
    i_Far_neg_o = pars.n_neg_o*F*(pars.k_fwd_star_neg_o*math.exp((-pars.beta_o*pars.n_neg_o*F*dphi_neg)/(R*pars.T))*pars.prod_fwd_neg_o
        -pars.k_rev_star_neg_o*math.exp(((1-pars.beta_o)*pars.n_neg_o*F*dphi_neg)/(R*pars.T)))
    
    i_Far_neg_p = pars.n_neg_p*F*(pars.k_fwd_star_neg_p*math.exp((-pars.beta_p*pars.n_neg_p*F*dphi_neg)/(R*pars.T))*pars.prod_fwd_neg_p
        -pars.k_rev_star_neg_p*math.exp(((1-pars.beta_p)*pars.n_neg_p*F*dphi_neg)/(R*pars.T)))
    
    i_Far_neg = i_Far_neg_o + i_Far_neg_p
    
    #Final calculations for the change in potential difference on the negatrode
    i_dl_neg = pars.i_ext - i_Far_neg
    ddphi_int_neg = -i_dl_neg/pars.C_dl_neg
    dSV_dt[ptr.dphi_int_neg] = ddphi_int_neg

    "Negatrode Gas Transport"
    #Getting parameters from the SV
    C_k_gd_neg = SV[ptr.C_k_gd_neg]
    C_k_rxn_neg = SV[ptr.C_k_gd_neg]
    #Making dictionaries for the gas diffusion equation:
    s1 = {'C_k':C_k_gd_neg,'dy':pars.dy_neg1}
    s2 = {'C_k':C_k_rxn_neg,'dy':pars.dy_neg2}
    gasProps = {'Kg':pars.Kg_neg,'t_fac':pars.tau_fac_neg,'D_k':pars.D_k_gas_neg, 'mu':pars.dyn_vis_gas,'T':pars.T}
    #Running gas transport function
    N_k_i = electrode_gas_transport(s1,s2,gasProps) #
    "Negatrode Gas Phase Reactions"
    #Hydrogen adsorption:
    prod_fwd_h2a = (C_k_rxn_neg[0]/np.sum(C_k_rxn_neg))**-ptr.nu_H2_gas_g * pars.C_vac_Ni**-pars.nu_vac_Ni_g #- signs are needed to cancel out the sign convention of the stoichiometric coefficients
    prod_rev_h2a = pars.C_H_Ni**pars.nu_H_Ni_g 
    qdot_h2a = pars.k_fwd_h2a * prod_fwd_h2a - pars.k_rev_h2a * prod_rev_h2a
    
    sdot_H2 = pars.nu_H2_gas_g * qdot_h2a #int not an array
    #Water desorption:
    prod_fwd_neg_wd = pars.C_H2O_Ni**pars.nu_H2O_Ni_g #- signs are needed to cancel out the sign convention of the stoichiometric coefficients
    prod_rev_neg_wd = (C_k_rxn_neg[2]/np.sum(C_k_rxn_neg))**-ptr.nu_H20_neg_g * pars.C_vac_Ni**pars.nu_vac_Ni_g
    qdot_h2a = pars.k_fwd_neg_wd * prod_fwd_neg_wd - pars.k_fwd_neg_wd * prod_rev_neg_wd
    
    sdot_H20_gas_neg = pars.nu_H2_gas_g * qdot_h2a

    #final gas phase equation
    sdot_k = np.array([sdot_H2,0,sdot_H20_gas_neg]) #hydrogen, oxygen, water
    #Need to find the area of the ni in the negatrode
    dCk_dt = (N_k_i + sdot_k*pars.A_fac_Pt)*pars.eps_g_dy_Inv_rxn
    dSV_dt[ptr.C_k_an_CL] = dCk_dt

    #----- Positrode ------------------------------------------------
    dphi_pos = SV[ptr.dphi_int_pos]
    
    #Mass-Action equations
    i_Far_pos_o = pars.n_pos_o*F*(pars.k_fwd_star_pos_o*math.exp((-pars.beta_o*pars.n_pos_o*F*dphi_pos)/(R*pars.T))*pars.prod_fwd_pos_o
        -pars.k_rev_star_pos_o*math.exp(((1-pars.beta_o)*pars.n_pos_o*F*dphi_pos)/(R*pars.T)))
    i_Far_pos_p = pars.n_pos_p*F*(pars.k_fwd_star_pos_p*math.exp((-pars.beta_p*pars.n_pos_p*F*dphi_pos)/(R*pars.T))*pars.prod_fwd_pos_p
        -pars.k_rev_star_pos_p*math.exp(((1-pars.beta_p)*pars.n_pos_p*F*dphi_pos)/(R*pars.T)))
    i_Far_pos = i_Far_pos_o + i_Far_pos_p
    
    i_dl_pos = pars.i_ext - i_Far_pos
    ddphi_int_pos = -i_dl_pos/pars.C_dl_pos
    dSV_dt[ptr.dphi_int_pos] = ddphi_int_pos
    return dSV_dt