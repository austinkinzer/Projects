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
t_final = 1000 #seconds

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

"Physical Constants:"
F = 96485 #C/mol e-
R = 8.314 #J/mol*K

"Equation values"
T = 823 #K

"Chemical parameters: (For these I just used yours, not sure where/how to find them) (I also kept hte positrode and negatrode values the same for now)"
#Negatrode ORR
k_fwd_star_neg_o = 4.16307062e+1 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_neg_o = 4.0650045e-1 #Chemical reverse rate constant, m^4/mol^2/s
#Negatrode HER also neeed to look these up, but im assuming they are much faster than the oxide ones
k_fwd_star_neg_p = 4.16307062e+3 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_neg_p = 4.0650045e+1 #Chemical reverse rate constant, m^4/mol^2/s
#Positrode OER
k_fwd_star_pos_o = 4.16307062e+1 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_pos_o = 4.0650045e-1 #Chemical reverse rate constant, m^4/mol^2/s
#Positrode HRR also neeed to look these up, but im assuming they are much faster than the oxide ones
k_fwd_star_pos_p = 4.16307062e+3 # Chemical forward rate constant, m^4/mol^2/s
k_rev_star_pos_p = 4.0650045e+1 #Chemical reverse rate constant, m^4/mol^2/s


"Material Parameters"
#BCZYYb4411 parameters:
ele_cond = 0.001 #1/(ohm*m) Need to look up this value so I just used yours
C_elyte = 46050    # Total (reference) elyte concentration, mol/m2 (I will calculate this at a later point)
D_k = np.array([7.46*10**-11,1.28*10**-12,0]) #(m^2/s) [Proton,Oxygen,Vacancy] Again I need to look these up so I used yours
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
#anode
eps_Ni = 0.455 #see calculations
eps_elyte_neg = 0.545 #See Calculations
r_Ni_neg = 1*10**-5 #(m)need to measure more accurately
r_elyte_neg = 5*10**-6 #(m) need to measure more acurately
r_int = 4*10**-6 #(m) need to measure more accurately, interface region between particles, on SEM images it looks almost like the radius
#Cathode
r_BCFZY = 500*10**-9 #(m) need to measure more accurately
eps_BCFZY = 0.5 #just assuming 50% porosity need to look up this value

"Thermodynamic values (first 5 taken from homework 4, last one I had to make up)"
g_H_Ni_o = -7.109209e+07      # standard-state gibbs energy for H adsorbed on Ni surface (J/kmol)
g_H2O_Ni_o = -3.97403035e+08  # standard-state gibbs energy for H2O adsorbed on Ni surface (J/kmol)
g_Vac_Ni_o = 0.0              # standard-state gibbs energy for Ni surface vacancy (J/kmol)
g_Vac_elyte_o = 0.0           # standard-state gibbs energy for electrolyte oxide vacancy (J/kmol)
g_Ox_elyte_o = -2.1392135e+08 # standard-state gibbs energy for electrolyte oxide O2- (J/kmol)
g_Hx_elyte_o = -2.1392135e+07 # standard-state gibbs energy for electrolyte protons H+ (J/kmol)

"Stoichiometric values:"
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


#/\/\/\/\/\ Initializing Solution Vector /\/\/\/\/\
SV_0 = np.array([dphi_int_neg_0 ,dphi_int_pos_0 ])

#/\/\/\/\/\ Making the parameter class /\/\/\/\/\
class pars:
    #important parameters
    i_ext = i_ext
    C_dl_neg = C_dl_neg
    time_span = np.array([0,t_final])
    T = T

    #beta values
    beta_o = 0.5 
    beta_p = 0.5

    #Interface Potentials
    dphi_int_neg_0 = phi_elyte_0-phi_neg_0 #Sets the applied potential on the cell\
    dphi_int_pos_0 = phi_pos_0-phi_elyte_0

    "Chemical parameters: (For these I just used yours, not sure where/how to find them) (I also kept hte positrode and negatrode values the same for now)"
    k_fwd_star_neg_o = k_fwd_star_neg_o
    k_rev_star_neg_o = k_rev_star_neg_o
    #Negatrode HER 
    k_fwd_star_neg_p =  k_fwd_star_neg_p
    k_rev_star_neg_p = k_rev_star_neg_p
    #Positrode OER
    k_fwd_star_pos_o = k_fwd_star_pos_o
    k_rev_star_pos_o = k_rev_star_pos_o
    #Positrode HRR 
    k_fwd_star_pos_p = k_fwd_star_pos_p
    k_rev_star_pos_p = k_rev_star_pos_p

    "Material Parameters"
    #BCZYYb4411 parameters:
    ele_cond = ele_cond
    C_elyte = C_elyte
    D_k = D_k
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
    A_surf_Ni_neg = 4*math.pi*r_Ni_neg**2
    A_surf_elyte_neg = 4*math.pi*r_elyte_neg**2
    #Cathode Geometric Parameters
    A_surf_BCFZY = 4*math.pi*r_BCFZY**2

    "Anode Product calculations" #Calculates the product terms in the mass action equations
    prod_fwd_neg_o = C_Ox_elyte**-nu_Ox_elyte_neg_o * C_H_Ni**-nu_H_Ni_neg_o  #- signs are needed to cancel out the sign convention of the stoichiometric coefficients
    prod_fwd_neg_p = C_H_Ni**-nu_H_Ni_neg_p * C_vac_Ni**-nu_vac_ely_neg_p
    prod_rev_neg_o = C_vac_elyte**nu_vac_elyte_neg_o * C_H2O_Ni**nu_H2O_Ni_neg_o * C_vac_Ni**nu_vac_Ni_neg_o
    prod_rev_neg_p = C_Hx_elyte**nu_Hx_ely_neg_p * C_vac_Ni**nu_vac_Ni_neg_p

#/\/\/\/\/\ Making the pointer class/\/\/\/\/\
#specifies where in SV certain variables are stored
class ptr:
    dphi_int_neg_0 = 0
    
    # C_k in anode GDL: starts just after phi_int_net_0
    dphi_int_pos_0 = np.arange(dphi_int_neg_0+1)


