# Import Libraries
import numpy as np 


# Functions 
def reynoldsNum(u,d,viscosity):
    # Reynolds Number   
    re_D = (u * d)/viscosity
    return re_D
def graetzNum(re_d,Pr,D,l):
    # Graetz Number
    gr = (re_d *Pr*D)/l
    return gr
def nusseltNum(gz):
    # Nusselt Number 
    #nu_d = (3.657)/(math.tanh((2.264*gz**(-1/3))+(1.7*gz**(-2/3)))+(0.0499*gz*math.tanh(gz**(-1))))
    nu_d = 3.657 + 0.2362*(gz**0.488)*np.exp(-57.2/gz)
    # Common Nusselt Numbers laminar flow [constant Wall temp, constant heat flux] = [3.657,4.364]
    return nu_d

#### Parameters
u = 0.5 # average velocity m/s
d = 0.00454 # pipe diameter in m 
length =  0.04860 # m distance along fluid flow direction, fully developed flow starts here 0.0394m
l_fdf = 0.045  #fully developed flow regime in m 
viscosity = 1.69e-5  # kinematic viscosity of air m^2/s
density =  1.09 # @ 50C of temperature at heated wall 100 C [kg/m^3] = 0.9467
thmCond = 0.0270# thermal conductivity of fluid at room temp 0.0270 W/mK
specific_heat = 1007 # J/kgK
T_wall = 50 # C
T_inlet = 20 #C, bulk temp
x = 0.05300 #length - l_fdf # this is the length starting at the casing where heating is applied

# compute the prandtl number so that we can incorporate the diffusivity 
# Pr = 0.709     # prandtl number for air at room temp
nu = 1.5e-5 #m^2/s kinematic viscosity
# alpha = 2.15e-5 # m^2/s thermal diffusivity of air at room temp
# descriptor = "Air at Room Temp"
# alpha = 2.0e-5
# descriptor = "Nickel"
alpha = 1.0e-8
descriptor = "diffusivity of air at 1e-8"
Pr = nu/alpha
print(f'Pr = {Pr}')

### Bulk Temperature Generation 
def graetz_bulkTemp(u,d,viscosity,Pr,length,thmCond,specific_heat,T_inlet):
    re_d = reynoldsNum(u,d,viscosity)
    gz = graetzNum(re_d,Pr,d,length)
    nu_d = nusseltNum(gz)
    h = nu_d* thmCond /d
    # Equation 7.59b comes from an energy balance equation
    temp_diff = 1 - np.exp(-(h/(density*specific_heat*u))*(4*length/d)) 
    T_b = T_inlet + (T_wall-T_inlet) *temp_diff 
    return re_d,gz,nu_d,h,T_b


re_d,gz,nu_d,h,T_b = graetz_bulkTemp(u,d,viscosity,Pr,x,thmCond,specific_heat,T_inlet)
print(f'Graetz {descriptor} ')
print(f'alpha = {alpha}')
print(f"Reynolds Number (Re_D): {re_d:.3f}")
print(f"Graetz Number (Gz): {gz:.3f}")
print(f"Nusselt Number (Nu_D): {nu_d:.3f}")
print(f"Heat Transfer Coefficient (h): {h:.3f}  W/m²K")
print(f"Bulk Temperature at x = {x*1000:.1f} mm: {T_b+273.15:.5f}°K")