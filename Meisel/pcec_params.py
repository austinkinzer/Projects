#Initilize variables for the PCEC model
#as of 30Nov20 most of these variables are hard coded and need to be found
#The solution vector is initialized here
#Everything here is saved into a pointer class

#\/\/^\/^\/\/^\ Imports /\/\/^\/^\/\/^\#
import numpy as np
import math


#\/\/^\/^\/\/^\ Parameters /\/\/^\/^\/\/^\#
"Importnat Variables For the SV to work"
i_ext = 40 #A
C_dl_neg = 6e5 # F/m2 this makes it so my function does not go to negative infinity
C_dl_pos = 2e2 # F/m2 (need to look up)
t_final = 1000 #seconds

"Physical Constants:"
F = 96485 #C/mol e-
R = 8.314 #J/mol*K

"Equation values"
T = 823 #K

"Initial Gas Concentrations/parameters"
#For now this only applies to the negatrode
#-----Negatrode: For now there is a non reactive node and a reaction node
#Initial Pressures
P_neg_gd = 101325 # Pa
P_neg_rxn = 81343 # Pa

#Node thicknesses:
dy1 = 980e-6 # m
dy2 = 20e-6 # m

# Initial mol fractions
X_k_gd = np.array([0.50, 0.50, 0.0]) #H2, N2, Steam
X_k_rxn = np.array([0.40, 0.55, 0.05]) #H2, N2, Steam

#Concentrations
C_k_gd_neg0 = X_k_gd*((P_neg_gd)/(R*T)) #H2, N2, Steam
C_k_rxn_neg0 = X_k_rxn*((P_neg_rxn)/(R*T)) #H2, N2, Steam

#Gas properties:
mu = 2.08e-5 #kg/m-s #I am going to use your value for this (dynamic viscosity)
D_k_an = np.array([0.3e-5, 2.798e-5, 1.9e-5]) #m2/s, H2, N2, Steam (gas diffusion, your values)

"n_values:"
n_neg_p = -1
n_neg_o = -2
n_pos_p = 2
n_pos_o = 2

"Potentials (I will place in more accurate numbers later) The anode is the reference"
phi_neg_0 = 0 #this will by my reference electrode
phi_elyte_0 = 0.5 # I will calculate more accurately later
phi_pos_0 = 1.05 # I will calculate more accurately later

dphi_int_neg_0 = phi_elyte_0-phi_neg_0 #Sets the applied potential on the cell\
dphi_int_pos_0 = phi_pos_0-phi_elyte_0

"beta values (need to also look up)"
beta_o = 0.5 
beta_p = 0.5

"Chemical parameters: (For these I just used yours, not sure where/how to find them) (I also kept hte positrode and negatrode values the same for now)"
#Negatrode OER
k_fwd_star_neg_o = 4.16307062e+1 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_neg_o = 4.0650045e-1 #Chemical reverse rate constant, m^4/mol^2/s
#Negatrode HER also neeed to look these up, but im assuming they are much faster than the oxide ones
k_fwd_star_neg_p = 4.16307062e+3 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_neg_p = 4.0650045e+1 #Chemical reverse rate constant, m^4/mol^2/s
#Positrode ORR
k_fwd_star_pos_o = 4.16307062e+1 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_pos_o = 4.0650045e-1 #Chemical reverse rate constant, m^4/mol^2/s
#Positrode HRR also neeed to look these up, but im assuming they are much faster than the oxide ones
k_fwd_star_pos_p = 4.16307062e+3 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_pos_p = 4.0650045e+1 #Chemical reverse rate constant, m^4/mol^2/s
#Negatrode water desorption: (water desorbing from Ni at 550C) (K^*=K, only a chemical reaction)
k_fwd_star_neg_wd = 1 #m^4/(mol^2*s) (need to look up)
k_rev_star_neg_wd = 1 #m^4/(mol^2*s) (need to look up)
#Negatrode Hydrogen adsorption (Hydrogen gas adsorbing from Ni at 550C)
k_fwd_star_h2a = 1 #m^4/(mol^2*s) (need to look up)
k_rev_star_h2a = 1 #m^4/(mol^2*s) (need to look up)


"Material Parameters"
#BCZYYb4411 parameters:
ele_cond = 0.001 #1/(ohm*m) Need to look up this value so I just used yours
C_elyte = 46050    # Total (reference) elyte concentration, mol/m2 (I will calculate this at a later point)
D_k_ely = np.array([7.46*10**-11,1.28*10**-12,0]) #(m^2/s) [Proton,Oxygen,Vacancy] Again I need to look these up so I used yours
#Nickle parameters:
C_Ni_s = 2.6e-05 #Surface site Concentrations mol/m^2 (again this is just from hw4)
#BCFZY parameters:
C_BCFZY = 46000 #mol/m^2 surface site concentration, I will look this up (If it is not known I will estimate it) likely it is similar to the elyte

"Concentrations/activities: I need to look these up so I used yours from HW4."
#-----Negatrode:
#Mol fractions (no units)
X_H_Ni = 0.6 #HW4
X_H2O_Ni = 0.2 #HW4
X_vac_Ni = 0.2 #HW4
X_Ox_elyte = 0.8 #I know this is 0.8
X_Hx_elyte = 0.1 #I am unsure of the split between Hx and oxygen vacancies
X_vac_elyte = 0.1 
#-----Positrode:
#Mol fractions (no units) #I made these up, all I know is that 80% of the lattice sites:
X_Hx_BF = 0.05
X_H2O_BF = 0.05
X_vac_BF = 0.05
X_O_BF = 0.05
X_Ox_BF = 0.8

"geometric parameters"
n_brugg = -0.5 #bruggman factor assuming alpha is -1.5
#anode
eps_Ni = 0.159 #see calculations
eps_elyte_neg = 0.191 #See Calculations
eps_gas_neg = 1-eps_Ni-eps_elyte_neg
d_Ni_neg = 1*10**-5 #(m)rough estimate from SEM images (average diameter of Ni in negatrode)
d_elyte_neg = 5*10**-6 #(m) rough estimate from SEM images (average diameter of BCZYYb in negatrode)
d_part_avg = (d_Ni_neg+d_elyte_neg)/2 #just taking a linear average of the two particle sizes
r_int = 2*10**-6 #(m) rough estimate from SEM images, interface region between particles, on SEM images it looks almost like the radius
#Cathode
d_BCFZY = 500*10**-9 #(m) rough estimate from SEM images
eps_BCFZY = 0.5 #just assuming 50% porosity need to look up this value could measure more accurately
eps_gas_pos = 1-eps_BCFZY

"Thermodynamic values (first 5 taken from homework 4, last one I had to make up)"
g_H_Ni_o = -7.109209e+07      # standard-state gibbs energy for H adsorbed on Ni surface (J/kmol)
g_H2O_Ni_o = -3.97403035e+08  # standard-state gibbs energy for H2O adsorbed on Ni surface (J/kmol)
g_Vac_Ni_o = 0.0              # standard-state gibbs energy for Ni surface vacancy (J/kmol)
g_Vac_elyte_o = 0.0           # standard-state gibbs energy for electrolyte oxide vacancy (J/kmol)
g_Ox_elyte_o = -2.1392135e+08 # standard-state gibbs energy for electrolyte oxide O2- (J/kmol)
g_Hx_elyte_o = -2.1392135e+07 # standard-state gibbs energy for electrolyte protons H+ (J/kmol)

"Stoichiometric values For the charge transfer reactions:"
#negatrode proton reaction:
nu_H_Ni_neg_p = -1
nu_vac_ely_neg_p = -1
nu_Hx_ely_neg_p = 1
nu_vac_Ni_neg_p = 1
#negatrode oxide reaction:
nu_H_Ni_neg_o = -2
nu_H2O_Ni_neg_o = 1
nu_vac_Ni_neg_o = 1
nu_vac_elyte_neg_o = 1
nu_Ox_elyte_neg_o = -1
#postirode proton reaction:
nu_Hx_BF_pos_p = -2
nu_O_BF_pos_p = -1
nu_H2O_BF_pos_p = 1
nu_vac_BF_pos_p = 1
#positrode oxide reaction:
nu_O_BF_pos_o = -1
nu_Ox_BF_pos_o = 1
nu_vac_BF_pos_o = 1

"Stoichiometric values for the gas transfer reactions at the negatrode"
#Hydrogen adsorption
nu_H_Ni_g = 2
nu_vac_Ni_g = -2
nu_H2_gas_g = -1
#Water desorption:
nu_H2O_Ni_g = -1
nu_vac_Ni_g = 1
nu_H20_gas_neg_g = 1

#/\/\/\/\/\ Initializing Solution Vector /\/\/\/\/\
SV_0 = np.array([dphi_int_neg_0, C_k_gd_neg0, C_k_rxn_neg0, dphi_int_pos_0 ])

#/\/\/\/\/\ Making the parameter class /\/\/\/\/\
class pars:
    #important parameters
    i_ext = i_ext
    C_dl_neg = C_dl_neg
    C_dl_pos = C_dl_pos
    time_span = np.array([0,t_final])
    T = T

    "Gas diffusion parameters"
    #Calculations
    tau_fac_neg = eps_gas_neg**n_brugg
    Kg_neg = (eps_gas_neg**3*d_part_avg**2)/(72*tau_fac_neg*(1-eps_gas_neg)**2)
    #Node thicknesses:
    dy_neg1 = dy1
    dy_neg2 = dy2
    eps_g_dy_Inv_rxn = 1/(dy2*eps_gas_neg)
    #Gas properties:
    dyn_vis_gas = mu 
    D_k_gas_neg = D_k_an

    #beta values
    beta_o = 0.5 
    beta_p = 0.5

    #Interface Potentials
    dphi_int_neg_0 = phi_elyte_0-phi_neg_0 #Sets the applied potential on the cell\
    dphi_int_pos_0 = phi_pos_0-phi_elyte_0

    "Chemical parameters: (For these I just used yours, not sure where/how to find them) (I also kept hte positrode and negatrode values the same for now)"
    #Negatrode OER
    k_fwd_star_neg_o = k_fwd_star_neg_o
    k_rev_star_neg_o = k_rev_star_neg_o
    #Negatrode HER 
    k_fwd_star_neg_p =  k_fwd_star_neg_p
    k_rev_star_neg_p = k_rev_star_neg_p
    #Positrode ORR
    k_fwd_star_pos_o = k_fwd_star_pos_o
    k_rev_star_pos_o = k_rev_star_pos_o
    #Positrode HRR 
    k_fwd_star_pos_p = k_fwd_star_pos_p
    k_rev_star_pos_p = k_rev_star_pos_p
    #Negatrode water desorption: (water desorbing from Ni at 550C)
    k_fwd_neg_wd = k_fwd_star_neg_wd
    k_rev_neg_wd = k_rev_star_neg_wd
    #Negatrode Hydrogen adsorption (Hydrogen gas adsorbing from Ni at 550C)
    k_fwd_h2a = k_fwd_star_h2a
    k_rev_h2a = k_rev_star_h2a

    "Material Parameters"
    #BCZYYb4411 parameters:
    ele_cond = ele_cond
    C_elyte = C_elyte
    D_k = D_k_ely
    #Nickle parameters:
    C_Ni_s = C_Ni_s
    #BCFZY parameters:
    C_BCFZY = C_BCFZY

    "Activity concentrations"
    #Negatrode Activity Concentrations: (mol/m^2)
    C_H_Ni = X_H_Ni*C_Ni_s
    C_H2O_Ni = X_H2O_Ni*C_Ni_s
    C_vac_Ni = X_vac_Ni*C_Ni_s
    C_Hx_elyte = X_Hx_elyte*C_elyte
    C_Ox_elyte = X_Ox_elyte*C_elyte
    C_vac_elyte = X_vac_elyte*C_elyte
    #Positrode Activity Concentrations: (mol/m^2)
    C_Hx_BF = X_Hx_BF*C_BCFZY
    C_H2O_BF = X_H2O_BF*C_BCFZY
    C_vac_BF = X_vac_BF*C_BCFZY
    C_O_BF = X_O_BF*C_BCFZY
    C_Ox_BF = X_Ox_elyte*C_BCFZY

    "Geometric parameters"
    #Anode Geometric Parameters
    L_TPB = 2*math.pi*r_int
    A_surf_Ni_neg = 4*math.pi*(d_Ni_neg/2)**2
    A_surf_elyte_neg = 4*math.pi*(d_elyte_neg/2)**2
    tau_fac_neg = eps_gas_neg**n_brugg #tortuosity factor
    Kg_neg = (eps_gas_neg**3*d_part_avg**2)/(72*tau_fac_neg*(1-eps_gas_neg)**2) #gas permeability, see calculations for more details
    #Cathode Geometric Parameters
    A_surf_BCFZY = 4*math.pi*(d_BCFZY/2)**2
    tau_fac_pos = eps_gas_pos**n_brugg
    Kg_pos = (eps_gas_pos**3*d_part_avg**2)/(72*tau_fac_pos*(1-eps_gas_neg)**2)

    "Negatrode Product calculations" #Calculates the product terms in the mass action equations
    prod_fwd_neg_o = C_Ox_elyte**-nu_Ox_elyte_neg_o * C_H_Ni**-nu_H_Ni_neg_o  #- signs are needed to cancel out the sign convention of the stoichiometric coefficients
    prod_fwd_neg_p = C_H_Ni**-nu_H_Ni_neg_p * C_vac_Ni**-nu_vac_ely_neg_p
    prod_rev_neg_o = C_vac_elyte**nu_vac_elyte_neg_o * C_H2O_Ni**nu_H2O_Ni_neg_o * C_vac_Ni**nu_vac_Ni_neg_o
    prod_rev_neg_p = C_Hx_elyte**nu_Hx_ely_neg_p * C_vac_Ni**nu_vac_Ni_neg_p

    "Positrode Product calculations" #Calculates the product terms in the mass action equations
    prod_fwd_pos_o = C_O_BF**-nu_O_BF_pos_o #- signs are needed to cancel out the sign convention of the stoichiometric coefficients
    prod_fwd_pos_p = C_Hx_BF**-nu_Hx_BF_pos_p * C_O_BF**-nu_O_BF_pos_p
    prod_rev_pos_o = C_Ox_BF**nu_Ox_BF_pos_o
    prod_rev_pos_p = C_H2O_BF**nu_H2O_BF_pos_p * C_vac_BF**nu_vac_BF_pos_p

    "Stoichiometric coefficients"
    #Hydrogen adsorption
    nu_H_Ni_g = nu_H_Ni_g
    nu_vac_Ni_g = nu_vac_Ni_g
    nu_H2_gas_g = nu_H2_gas_g
    #Water desorption:
    nu_H2O_Ni_g = nu_H2O_Ni_g
    nu_vac_Ni_g = nu_vac_Ni_g
    nu_H20_neg_g = nu_H20_gas_neg_g
    
#/\/\/\/\/\ Making the pointer class/\/\/\/\/\
#specifies where in SV certain variables are stored
class ptr:
    dphi_int_neg = 0
    
    C_k_gd_neg = 1

    C_k_rxn_neg = np.arange(C_k_gd_neg+1)
    
    dphi_int_pos = 3


