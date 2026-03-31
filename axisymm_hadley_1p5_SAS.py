"""
This module calculates a semi-analytical solution to the 1.5-layer model
of the Hadley circulation using the shooting method throughout the
ascending branch, as described in Eusebi and Schneider (2026).
"""


from scipy.optimize import fsolve
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from numpy import sin, cos, tan
from scipy.integrate import quad
import xarray as xr
from re_profiles import re_multi_plevel
from scipy.interpolate import interp1d

"""
Constants
"""
drad = np.pi/180
Rd = 287 # Specific gas constant for dry air in J/kg/K
cp = 1005 # Specific heat capacity at constant pressure for dry air in J/kg/K
g = 9.81 # Acceleration due to gravity in m/s^2


def get_vma_dtheta(M, phi, C_theta,omega, a):
   """
   Get meridional gradient of theta for calculation of theta
   using shooting method
   Inputs:
      M: angular momentum in m^2/s
      phi: latitude in radians
      C_theta: theta conversion factor for vertically integrated theta
      omega: planetary rotation rate in rad/s
   Outputs:
      dtheta: meridional gradient of theta in K/m
   """
   alpha = -C_theta*Rd
   u = (M - omega*a**2*cos(phi)**2)/(a*cos(phi))
   dtheta = 1/alpha * (2*omega*a*sin(phi)*u + tan(phi)*u**2)
   return dtheta


def get_theta_from_M(M0, theta0, phi, phi_b, C_theta, omega, a):
   """
   Calculate classic AMC theta profile corresponding to constant angular momentum,
   valid in the 1.5-layer model solution poleward of phi_b, the latitude separating the 
   descending and ascending branches (where VMA no longer alters the upper branch angular momentum balance).
   Inputs:
      M0: angular momentum at phi_b in m^2/s
      theta0: theta at phi_b in K
      phi: latitude in radians
      phi_b: latitude of phi_b in radians
      C_theta: theta conversion factor for vertically integrated theta
      omega: planetary rotation rate in rad/s  
   Outputs:
      theta: xarray DataArray with theta profile in K
   """
   alpha = -C_theta*Rd
   theta = phi*0 + theta0
   int_ref = omega**2 * a**2 * cos(2*phi_b)/(4*alpha)  +  \
            1/alpha/(2*a**2*cos(phi_b)**2)*M0**2
   int = omega**2 * a**2 * cos(2*phi)/(4*alpha)  +  \
            1/alpha/(2*a**2*cos(phi)**2)*M0**2

   theta = theta + int - int_ref

   return theta

def get_u_from_theta(theta, C_theta,omega, a):
   """
   Calculate u from theta through thermal wind balance
   Inputs:
      theta: xarray DataArray with theta profile in K
      omega: planetary rotation rate in rad/s
      a: Earth's radius in m
   Outputs:
      u: xarray DataArray with u profile in m/s
   """
   alpha = -C_theta*Rd
   dtheta_dphi = theta.differentiate('lat')/drad
   cosphi = cos(np.radians(theta.lat))
   sinphi = sin(np.radians(theta.lat))
   u = omega*a*cosphi*( np.sqrt( 1 + alpha/(omega**2 * a**2 * sinphi*cosphi)*dtheta_dphi ) - 1 )
   return u

# thetae should have coords in radians
# phi_m should be in radians
# dphi should be in radians
def get_SAS_theta(thetae, phi_m, thetam, dphi, C_theta, omega, a, rce_func=None):
   """ 
   Calculate theta profile for 1.5-layer model of Hadley circulation 
   for phi_a = phi_m and theta_a = thetam using the shooting method.
   Inputs:
      thetae: xarray DataArray with RCE profile of theta in K
      phi_m: latitude of phi_a in radians
      thetam: theta at phi_m in K
      dphi: latitude spacing in radians
      C_theta: theta conversion factor for vertically integrated theta
      omega: planetary rotation rate in rad/s
      rce_func: function of latitude returning RCE profile of theta in K
   Outputs:
      theta: xarray DataArray with theta profile in K
      M: xarray DataArray with meridional profile of angular momentum in units m^2/s
      heat_int: xarray DataArray with heat integral in J/m^2
   """
   lat_max = np.radians(85)
   lat_min = -lat_max
   phi = np.arange(lat_min,lat_max+dphi/2,dphi)

   # Adjust phi to align with phi_m
   phi_m_idx = np.argmin(np.abs(phi-phi_m))
   phi = phi + (phi_m - phi[phi_m_idx])
   i_init = phi_m_idx
   if rce_func is not None:
      thetae = rce_func(phi)
   else:
      thetae = thetae.interp(lat=phi).values
   i_min = 0
   i_max = len(phi)
   theta = phi*0
   M = phi*0
   heat_int = phi*0

   # initialize theta and M at phi_m
   theta[i_init] = thetam
   M[i_init] = omega*a**2*cos(phi_m)**2
   heat_int[i_init] = (thetae[i_init] - theta[i_init])*cos(phi[i_init])

   # Apply shooting method to region north of phi_m:
   for i in range(i_init, i_max-1):      
      dtheta = get_vma_dtheta(M[i], phi[i], C_theta, omega, a)
      theta[i+1] = theta[i] + dtheta*dphi
      heat = (thetae[i+1] - theta[i+1])*cos(phi[i+1])
      heat_int[i+1] = heat_int[i] + heat
      Mp = omega*a**2*cos(phi[i+1])**2
      M[i+1] = (M[i]*heat_int[i] + Mp*heat)/heat_int[i+1]

      # Once poleward of phi_b, use AMC theta profile for rest of circulation
      # Will later correct profile poleward of phi_s, phi_n to RCE profile
      if theta[i+1] > thetae[i+1]:
         theta[i+2:] = get_theta_from_M(M[i+1], theta[i+1], phi[i+2:], phi[i+1], C_theta, omega, a)
         M[i+2:] = M[i+1]
         break

   # Apply shooting method to region south of phi_m
   for i in range(i_init, i_min+1, -1):
      dtheta = get_vma_dtheta(M[i], phi[i], C_theta, omega, a)
      theta[i-1] = theta[i] - dtheta*dphi
      heat = (thetae[i-1] - theta[i-1])*cos(phi[i-1])
      heat_int[i-1] = heat_int[i] + heat
      Mp = omega*a**2*cos(phi[i-1])**2
      M[i-1] = (M[i]*heat_int[i] + Mp*heat)/heat_int[i-1]

      # Once poleward of phi_b, use AMC theta profile for rest of circulation
      # Will later correct profile poleward of phi_s, phi_n to RCE profile
      if theta[i-1] > thetae[i-1]:
         theta[:i-1] = get_theta_from_M(M[i-1], theta[i-1], phi[:i-1], phi[i-1], C_theta, omega, a)
         M[:i-1] = M[i-1]
         break

   theta = xr.DataArray(theta, coords={'lat': ('lat', phi)})
   M = xr.DataArray(M, coords={'lat': ('lat', phi)})
   heat_int = xr.DataArray(heat_int, coords={'lat': ('lat', phi)})

   return theta, M, heat_int




def calc_heating_rate(phi, theta, tau, rce_func, rce_args):
   """
   Calculate vertically and zonally averaged heating rate at 
   latitude phi given theta profile and RCE profile
   Inputs:
      phi: latitude in radians
      theta: xarray DataArray with theta profile in K
      tau: function returning tau in seconds at latitude phi
      rce_func: function returning RCE profile of theta in K at latitude phi
      rce_args: tuple of arguments to pass to rce_func (could be None)
   Outputs:
      Q: heating rate in K/s
   """
   th_equil = rce_func(phi, *rce_args)
   th = theta(phi)
   Q = (th_equil - th)*cos(phi)/tau(phi)
   return Q

def edge_continuity(phi_s, phi_n, theta, rce_func, rce_args):
   amc_pot_se = theta(phi_s)
   amc_pot_ne = theta(phi_n)

   th_equil_se = rce_func(phi_s, *rce_args)
   th_equil_ne = rce_func(phi_n, *rce_args)

   return amc_pot_se - th_equil_se, amc_pot_ne - th_equil_ne


def tau_const(taus=50):
   """
   Calculate constant tau profile
   Inputs:
      taus: radiative relaxation timescale in days, default 50 days
   Outputs:
      tau: function of latitude returning tau in seconds
   """
   def tau(phi):
      return taus*86400
   return tau

def tau_gauss(phi0):
   """
   Calculate Gaussian tau profile with std 20 degrees and maximum at phi0
   Inputs:
      phi0: latitude of maximum tau in radians
   Outputs:
      tau: function of latitude returning tau in seconds
   """
   def tau(phi):
      tau_w = np.exp(-(phi-phi0)**2/(2*np.radians(20)**2)) 
      tau = tau_w*7 + (1-tau_w)*50
      tau *= 86400
      return tau
   return tau

   
# Calculate deviations from energy cons dervation within each Hadley cell
# Also ensure continuity of temperature profile with RCE profile at 
# Hadley cell edges.
# Returns array with quantified deviations of each of these 4 constraints
def conserve_energy(x, tau, C_theta, omega, a, rce_func, rce_args):
   """
   Calculate deviations from energy conservation within each Hadley cell
   and ensure continuity of temperature profile with RCE profile at
   Hadley cell edges.
   Inputs:
      x: array of initial conditions [phi_a, phi_n, phi_s, theta_a]
      tau: function returning tau in seconds at latitude phi
      C_theta: theta conversion factor for vertically integrated theta
      omega: planetary rotation rate in rad/s
      a: Earth's radius in m
      rce_func: function returning RCE profile of theta in K at latitude phi
      rce_args: tuple of arguments to pass to rce_func (could be None)
   Outputs:
      z: array of deviations from energy conservation and theta continuity
   """
   phi_a, phi_n, phi_s, theta_a = x

   dphi = np.radians(0.005)

   phi_n = min(phi_n, 90*drad)
   phi_s = max(phi_s, -90*drad)
   phi_a = min(phi_a, 90*drad)
   phi_a = max(phi_a, -90*drad)
   
   theta, _, _ = get_SAS_theta(None, phi_a, theta_a, dphi, C_theta, omega, a, rce_func)
   theta = interp1d(theta.lat.values, theta.values, bounds_error=False, fill_value="extrapolate")

   z1 = quad(calc_heating_rate, phi_a, phi_n, args=(theta, tau, rce_func, rce_args), epsrel=1e-16)[0]
   z2 = quad(calc_heating_rate, phi_s, phi_a, args=(theta, tau, rce_func, rce_args), epsrel=1e-16)[0]
   z3, z4 = edge_continuity(phi_s, phi_n, theta, rce_func, rce_args)
   z = np.array([z1, z2, z3, z4])
   print(z)
   
   return z


def hadley_solver(phi0, dh, dp, dz, C_theta, omega_factor=1, taus='const', rce_func=None, rce_args=None):
   """
   Calculate the semi-analytical solution to the 1.5-layer model of the Hadley circulation for 
   a given phi0. Solution optimizer is sensitive to initial conditions.
   Inputs:
      phi0: latitude of maximum RCE theta in radians
      dh: surface RE theta difference between equator and pole in K
      dp: pressure thickness of troposphere in Pa
      dz: potential temperature difference between lower and upper layers in K
      C_theta: theta conversion factor for vertically integrated theta
      omega_factor: factor by which to multiply Earth's rotation rate (default 1)
      taus: code for functional form of tau: 'const' for constant tau (default), 'gauss' for Gaussian tau
      rce_func: function returning RCE profile of theta in K at latitude phi
      rce_args: tuple of arguments to pass to rce_func (could be None)
   Outputs:
      ds: xarray Dataset with various circulation details
   """
   
   th0 = 290 # Initial guess of theta at phi_a
   a = 6.371e6 # Earth's radius in m

   # Get function for meridional profile of tau
   if taus == 'const':
      tau = tau_const()
   elif taus == 'gauss':
      tau = tau_gauss(phi0)

   # Calculate planetary rotation rate
   omega_E = 7.292e-5 # Earth's rotation rate in rad/s
   omega = omega_E*omega_factor

   phi_a = 0
   phi_n = 40*drad
   phi_s = -phi_n

   # Set initial conditions based on phi0 value
   # Ability of solver to find optimal solution is sensitive to initial condition
   # Three example initial conditions are provided for constant tau profiles
   # for phi0 of 0, 10, and 20 degrees. The user might wish to perform a grid 
   # search over a range of initial conditions to ensure convergence to the global minimum
   if taus == 'const':
      if np.degrees(phi0)<0.1:
         phi_n = 40*drad
         phi_s = -phi_n
         phi_a = 0
      elif round(np.degrees(phi0)) <11:
         phi_n = 40*drad
         phi_a = 25*drad
         phi_s = -phi_n
      elif round(np.degrees(phi0)) <21:
         phi_n = 35*drad
         phi_a = 30*drad
         phi_s = -phi_n

   x0 = [phi_a, phi_n, phi_s, th0]
   x = fsolve(conserve_energy, x0, args=(tau, C_theta, omega, a, rce_func, rce_args), epsfcn=1e-10, xtol=1e-16)
   phi_a, phi_n, phi_s, theta_a = x    
   print(f'RESULT for phi0={np.round(np.degrees(phi0),1)}: ', np.degrees(phi_a), np.degrees(phi_n), np.degrees(phi_s), theta_a) 


   # Get and save final profile outputs
   phi = np.linspace(-89, 89, 501)*drad
   dphi = np.radians(0.01)
   theta, M, heating_rate = get_SAS_theta(None, phi_a, theta_a, dphi, C_theta, omega, a, rce_func)
   
   theta = theta.interp(lat=phi).assign_coords(lat=np.degrees(phi))
   M = M.interp(lat=phi).assign_coords(lat=np.degrees(phi))
   heating_rate = heating_rate.interp(lat=phi).assign_coords(lat=np.degrees(phi))
   u = (M - omega*a**2*cos(phi)**2)/(a*cos(phi))
   thetae = rce_func(phi, *rce_args)

   # Theta profile outside Hadley cell boundaries are RCE profile
   theta[phi<phi_s] = thetae[phi<phi_s]
   theta[phi>phi_n] = thetae[phi>phi_n]
   u_thermalwind = get_u_from_theta(theta, C_theta, omega, a)
   u[phi<phi_s] = u_thermalwind[phi<phi_s]
   u[phi>phi_n] = u_thermalwind[phi>phi_n]
   heating_rate = (thetae - theta)*cos(phi)/tau(phi)

   theta_for_quad = interp1d(np.radians(theta.lat.values), theta.values, bounds_error=False, fill_value="extrapolate")
   heat_flux_cos = np.zeros(len(phi))

   # Calculate heat transport by Hadley circylation at all latitudes
   # by integration of heating profile from phi_a
   for k, phik in enumerate(phi):
      if phik>phi_s and phik<phi_n:
         heat_flux_cos[k] = quad(calc_heating_rate, phi_a, phik, args=(theta_for_quad, tau, rce_func, rce_args))[0]

   # Zonally and vertically integrated heat flux in units W
   heat_flux_cos = heat_flux_cos * (dp/g*a*2*np.pi*a) * cp
   mass_flux_cos = -heat_flux_cos / dz / cp

   print(f"phi0={np.degrees(phi0)}: ", np.degrees(phi_a), np.degrees(phi_n), np.degrees(phi_s), theta_a)

   # Construct dataset with circulation profiles and variables
   phi = np.degrees(phi)
   u = xr.DataArray(u, dims=['lat'], coords = {'lat':phi}).rename('u')
   theta = xr.DataArray(theta, dims=['lat'], coords = {'lat':phi}).rename('theta')
   thetae = xr.DataArray(thetae, dims=['lat'], coords = {'lat':phi}).rename('thetae')
   heatflux = xr.DataArray(heat_flux_cos, dims=['lat'], coords = {'lat':phi}).rename('heatflux')
   heating_rate = xr.DataArray(heating_rate, dims=['lat'], coords = {'lat':phi}).rename('heating_rate')
   psi = xr.DataArray(mass_flux_cos, dims=['lat'], coords = {'lat':phi}).rename('psi')
   phi_a = xr.DataArray(np.degrees(phi_a)).rename('phi_a')
   phi_n = xr.DataArray(np.degrees(phi_n)).rename('phi_n')
   phi_s = xr.DataArray(np.degrees(phi_s)).rename('phi_s')
   
   ds = xr.merge([u, theta, thetae, heatflux, heating_rate, psi, phi_a, phi_n, phi_s])
   return ds

def get_phi0_var(dh, dp, dz, tau, omega_factor=1):
   """
   Calculate the semi-analytical solution to the 1.5-layer model of the Hadley circulation for 
   a range of phi0 values.
   Inputs:
      dh: surface RE theta difference between equator and pole in K
      dp: pressure thickness of troposphere in Pa
      dz: potential temperature difference between lower and upper layers in K
      tau: code for functional form of tau: 'const' for constant tau (default), 'gauss' for Gaussian tau
      omega_factor: factor by which to multiply Earth's rotation rate (default 1)
   Outputs:
      ds: xarray Dataset with various circulation details for each phi0
   """
   ps = 1000 # Surface pressure in hPa
   pt = ps - dp/100 # Tropopause pressure in hPa
   kappa = Rd / cp
   C_theta = (ps**kappa - pt**kappa) / (kappa * ps**kappa)
   phi0s = np.array([0,10,20])
   ds = []
   for i, phi0 in enumerate(np.radians(phi0s)):
      print(phi0)
      phi = np.linspace(-90, 90, 1001)*drad

      # Get theta3 profile for given value of phi0
      thetae = re_multi_plevel(np.degrees(phi), np.degrees(phi0), 120, 310, pmin=pt, pmax=ps)
      thetae = np.mean(thetae, axis=1)      
      # Pass in thetae as an interpretable function of latitude
      rce_func = interp1d(phi, thetae, bounds_error=False, fill_value="extrapolate")

      # Solve for Hadley circulation for given value of phi0
      dsi = hadley_solver(phi0, dh, dp, dz, C_theta, omega_factor=omega_factor, taus=tau, rce_func=rce_func, rce_args=())
      dsi = dsi.expand_dims({'phi0': [phi0s[i]]})
      ds.append(dsi)

   ds = xr.concat(ds, dim='phi0')

   # Save to netcdf if desired
   # ds.to_netcdf()

   return ds

    

if __name__ == '__main__':
   dh = 120 # sureface RE theta difference between equator and pole in K
   dp = 800e2 # Pressure thickness of troposphere
   dz = 30 # Potential temperature difference between lower and upper layers in K
   tau = 'const'
   omega_factor = 1.0 # Omega factor relative to Earth's rotation rate
   ds = get_phi0_var(dh, dp, dz, tau, omega_factor)