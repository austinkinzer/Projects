"""
    This file runs and executes a your model, calculating the cell/device/system properties of interest as a function of time. 

    The code structure at this level is meant to be very simple, and delegates most functions to lower-level modules, which can be updated with new capabilties, over time.  The code:

        1 - Reads inputs and initializes the model
        
        2 - Calls the residual function and integrates over the user-defined    
            time span.
        3 - The simulation then returns the solution vector at the end of the 
            integration, which can be processed as needed to plot or analyze any quantities of interest.
"""

# Import necessary modules:
from scipy.integrate import solve_ivp #integration function for ODE system.
import numpy as np
from math import exp

# Constants
F = 96485 # Faraday's constant; C/mol equiv
R = 8.3145 # Gas constant; J/mol-K
pi = 3.141592653589793238462643

# Inputs
T = 273.15 + 75.00  # Temperature of 75 degrees C (based on mid-point of 60-90 deg C operating range listed in Fuller & Harb)

P_an_0 = 101325                   # initial pressure in anode in Pascals (atmospheric pressure)
X_k_an_0 = np.array([0.97, 0.03]) # initial mol fraction air and H2O in anode

i_ext = 10000 # External current in A (assumed constant)
t_f = 60000   # End time of simulation (number of time steps calculated)
time_span = np.array([0, t_f])

eps_g_GDL = 0.7  # Volume fraction of gas phase in gas diffusion layer
eps_s_CL = 0.6   # Volume fraction of solids in catalyst layer (carbon & platinum)
eps_g_CL = 0.28  # Volume fraction of gas in catalyst layer

# tau_fac = eps^brugg_m
brugg_GDL = -1.0  # Bruggeman correlation exponent in GDL
brugg_CL = -0.5 # Bruggeman correlation exponent in CL

# different fractions of carbon surface covered by Pt in CL to be analyzed.
#Pt_surf = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]) 
Pt_surf = 0.10

dy_GDL = 200*10**-6   # thickness of gas diffusion layer in m (Fuller & Harb) 
dy_CL = 4*10**-6      # thickness of catalyst layer in m (Darab et al.)

d_part_GDL = 12*10**-6 # carbon particle diameter in GDL in m (Benziger et al.)
d_part_CL = 50*10**-9  # carbon particle diameter in CL in m (Holdcroft)

D_k_g_an = np.array([5.48*10**-4, 6.13*10**-5]) # Average diffusion coefficients, m^2/s
mu_g_an = 9.54*10**-6                           # Dynamic viscosity, Pa-s

phi_an_0 = 0.0       # initial potential in anode (reference)
phi_elyte_0 = 0.6  # initial potential in electrolyte
phi_ca_0 = 1.1     # initial potential in cathode

C_dl_an = 1*10**2 # Capacitance of anode double layer F/m^2
C_dl_ca = 1*10**2 # Capacitance of cathode double layer F/m^2

i_o_an = 2.5*10**-3  # exchange current density in anode A/m^2
i_o_ca = 1*10**-3    # exchange current density in cathode A/m^2

n_an = -2.0 # charge equivalent transferred by rxn in anode
n_ca = 4.0    # charge equivalent transferred by rxn in cathode

nu_k_an = np.array([-1.0, 0.0])

alpha_ca = 0.5
alpha_an = 0.5

d_phi_eq_an = -0.61
d_phi_eq_ca = 0.55

# Initial concentration of species in catalyst layer
C_k_an_CL_0 = P_an_0*X_k_an_0/R/T


# Initial Solution Vector
SV_0 = np.hstack((np.array([phi_an_0 - phi_elyte_0]), C_k_an_CL_0, C_k_an_CL_0,
    np.array([phi_ca_0 - phi_elyte_0])))

# Price of Pt
price_Pt = 33000    # approximate price of platinum in $/kg (bullionbypost.com)

# Dimensions of Cell
length = 0.25    #planform length in m (Wheeler & Sverdrup, NREL)
width = 0.18     #planform width in m (Wheeler & Sverdrup, NREL)
    
# Diameter of Platinum particles in CL
d_Pt = 2*10**-9    #particle diameter in m (Darab et al., 2017)

# density of platinum
rho_Pt = 21450    #density of platinum in kg/m^3


#class structure containing all parameters
class pars:
    time_span = np.array([0,t_f])

    T = T

    i_ext = i_ext

    # Anode
    d_phi_eq_an = d_phi_eq_an
    i_o_an = i_o_an
    n_an = n_an
    alpha_an = alpha_an

    C_dl_an = C_dl_an

    dy_GDL = dy_GDL
    dy_CL = dy_CL

    inv_eps_dy_CL = 1/dy_CL/eps_g_CL
    inv_eps_dy_GDL = 1/dy_GDL/eps_g_GDL

    eps_g_GDL = eps_g_GDL
    eps_g_CL = eps_g_CL
    eps_s_CL = eps_s_CL
    brugg_GDL = brugg_GDL
    brugg_CL = brugg_CL

    nu_k_an = nu_k_an

    X_k_GDL = X_k_an_0

    D_k_g_an = D_k_g_an
    mu_g_an = mu_g_an

    Pt_surf = Pt_surf    # Fraction of carbon surface area covered by platinum

    
    A_fac_Pt = 0.5*eps_s_CL*3.*dy_CL*Pt_surf/d_part_CL # Platinum surface area factor
    A_fac_dl = Pt_surf/A_fac_Pt                            # Double Layer surface area factor

   
    d_s_GDL = d_part_GDL
    d_s_CL = d_part_CL
    
    # Cathode
    d_phi_eq_ca = d_phi_eq_ca
    i_o_ca = i_o_ca
    n_ca = n_ca
    alpha_ca = alpha_ca

    C_dl_ca = C_dl_ca
    
    # Platinum
    price_Pt = price_Pt    # $/kg
    
    length = length
    width = width
    area = length*width # cell area in m^2
    
    d_Pt = d_Pt
    rho_Pt = rho_Pt
    
#pointer function to denote locations of certain variables within solution vector
class ptr:
    phi_dl_an = 0
    
    # C_k in anode GDL: starts just after phi_dl, is same size as X_k_an:
    C_k_an_GDL = np.arange(phi_dl_an+1, phi_dl_an+1+X_k_an_0.shape[0])
    
    # C_k in anode CL: starts just after GDL, is same size as X_k_an:
    C_k_an_CL = np.arange(C_k_an_GDL[-1]+1, C_k_an_GDL[-1]+1+X_k_an_0.shape[0])
    
    # phi_dl_ca: starts just after C_k_an_CL:
    phi_dl_ca = C_k_an_CL[-1] + 1
  
    
def residual(t, SV, pars, ptr):
    # intialize matrix for residual
    dSV_dt = np.zeros_like(SV)

    # Anode
    
    # Calculate overpotential (eta)
    eta_an = SV[ptr.phi_dl_an] - pars.d_phi_eq_an 

    # Butler-Volmer formulation for Faradaic current:
    i_Far_an = -pars.i_o_an*(exp(-pars.n_an*F*pars.alpha_an*eta_an/R/pars.T)
                      - exp(pars.n_an*F*(1-pars.alpha_an)*eta_an/R/pars.T))

    # Current density at double layer
    i_dl_an = -pars.i_ext*pars.A_fac_dl - i_Far_an*pars.Pt_surf

    # Change in double layer potential per unit time
    dSV_dt[ptr.phi_dl_an] = -i_dl_an/pars.C_dl_an

    # Concentration of gas phase in diffusion layer
    C_k_an_GDL = SV[ptr.C_k_an_GDL]
    
    # Concentration of gas phase in catalyst layer
    C_k_an_CL = SV[ptr.C_k_an_CL]
   
    s1 = {'C_k': C_k_an_GDL, 'dy':pars.dy_GDL, 'eps_g':pars.eps_g_GDL, 
        'brugg':pars.brugg_GDL, 'd_s':pars.d_s_GDL}
    s2 = {'C_k': C_k_an_CL, 'dy':pars.dy_CL, 'eps_g':pars.eps_g_CL,
        'brugg':pars.brugg_CL, 'd_s':pars.d_s_CL}
    g_props = {'T':pars.T, 'D_k':pars.D_k_g_an, 'mu':pars.mu_g_an}
    cost_vars = {'area':pars.area, 'price_Pt':pars.price_Pt, 'eps_s':pars.eps_s_CL, 'd_Pt':pars.d_Pt,'rho':pars.rho_Pt}
    
    # Molar production rates from i_far:
    s_dot_k = i_Far_an*pars.nu_k_an/pars.n_an/F
    
    N_k_i = pemfc_gas_flux(s1, s2, g_props)
    
    # Change in gas mole fractions in catalyst per unit time:
    dCk_dt = (N_k_i + s_dot_k*pars.A_fac_Pt)*pars.inv_eps_dy_CL
    dSV_dt[ptr.C_k_an_CL] = dCk_dt

    
    # Cathode
    
    # Calculate overpotential
    eta_ca = SV[ptr.phi_dl_ca] - pars.d_phi_eq_ca
    i_Far_ca = pars.i_o_ca*(exp(-pars.n_ca*F*pars.alpha_ca*eta_ca/R/pars.T)
                      - exp(pars.n_ca*F*(1-pars.alpha_ca)*eta_ca/R/pars.T))
    i_dl_ca = pars.i_ext*pars.A_fac_dl - i_Far_ca*pars.Pt_surf
    
    
    dSV_dt[ptr.phi_dl_ca] = -i_dl_ca/pars.C_dl_ca
    return dSV_dt

def pemfc_gas_flux(s1, s2, g_props):
    N_k  = np.zeros_like(s1['C_k'])

    f1 = s1['dy']/(s1['dy'] + s2['dy'])
    f2 = 1-f1

    C_int = f1*s1['C_k'] + f2*s2['C_k']

    X_k_1 = s1['C_k']/np.sum(s1['C_k'])
    X_k_2 = s2['C_k']/np.sum(s2['C_k'])
    X_k_int = f1*X_k_1 + f2*X_k_2

    P_1 = np.sum(s1['C_k'])*R*(g_props['T'])
    P_2 = np.sum(s2['C_k'])*R*(g_props['T'])

    eps_g = f1*s1['eps_g'] + f2*s2['eps_g']
    tau_fac = (f1*s1['eps_g']**s1['brugg'] 
        + f2*s2['eps_g']**s2['brugg'])
    D_k_eff = eps_g*g_props['D_k']/tau_fac
    
    d_part = f1*s1['d_s'] + f2*s2['d_s']
    K_m = eps_g**3*d_part**2/(tau_fac**(2))/((1-eps_g)**(2))/72

    dY = 0.5*(s1['dy'] + s2['dy'])

    V_conv = -K_m*(P_2 - P_1)/dY/g_props['mu']
    V_k_diff = -D_k_eff*(X_k_2 - X_k_1)/dY/X_k_int

    V_k  = V_conv + V_k_diff

    N_k = C_int*X_k_int*V_k

    return N_k

def pemfc_pt_cost(pars):
            
    # array of surface fractions of Pt to be considered from 0.01 to 0.20
    Pt_surf_test = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40])
    # Pt_surf = 0.1
    
    #re-entering inputs to solve attribute error
   
    eps_s_CL = pars.eps_s_CL
    dy_CL = pars.dy_CL 
    d_s_CL = pars.d_s_CL
    
    # Price of Pt
    price_Pt = pars.price_Pt    # approximate price of platinum in $/kg (bullionbypost.com)

    # Dimensions of Cell
    length = pars.length   #planform length in m (Wheeler & Sverdrup, NREL)
    width = pars.width     #planform width in m (Wheeler & Sverdrup, NREL)
    area = length*width    #planform area
    
    # Diameter of Platinum particles in CL
    d_Pt = pars.d_Pt    #particle diameter in m (Darab et al., 2017)

    # density of platinum
    rho_Pt = pars.rho_Pt    #density of platinum in kg/m^3
    
    #calculate volume of solids in catalyst layer, including both platinum and carbon
    Vol_solids_CL = eps_s_CL*dy_CL*area
    
    #calculate number of carbon particles and number of platinum particles
    n_carbon = Vol_solids_CL/((4/3*pi*(d_s_CL/2)**3)+(4/3*pi*(d_Pt/2)*Pt_surf_test*4*(d_s_CL/2)**2))
    n_Pt = n_carbon*4*Pt_surf_test*(d_s_CL/2)**2/((d_Pt/2)**2)
    
    #calculate volume of platinum in m^3
    Vol_Pt = n_Pt*4/3*pi*(d_Pt/2)**3
    
    #calculate mass of platinum in kg
    m_Pt = Vol_Pt*rho_Pt
    
    #calculate cost of platinum in $
    cost = m_Pt*price_Pt

    return cost
  
