"""
Axisymmetric 1.5-Layer Hadley Circulation Model

This module implements a numerical model of the axisymmetric Hadley circulation
based on a 1.5-layer framework which is used in Eusebi & Schneider (2026). 
It is nearly identical to the model described in Held & Hou (1980) and 
Schneider & Lindzen (1988), but it includes the vertical momentum advection term 
in the zonal momentum equation. The code for this model is based on code written
by Timothy Cronin for the model described in Sobel & Schneider (2009).

References:
----------
- Eusebi & Schneider (2026)
- Held & Hou (1980)
- Schneider & Lindzen (1988)
- Sobel & Schneider 2009


Author: Ryan Eusebi
Date: 2026-02-26
"""

import numpy as np
from numpy import sin, cos, pi, tan
import xarray as xr
import argparse
from multiprocessing import Pool
import sys
sys.path.append('/resnick/groups/esm/reusebi/thesis/1D_HadleyModel')
from re_profiles import re_multi_plevel

Re = 6356000
g = 9.81
cp = 1005  # specific heat at constant pressure for dry air


def run_hadley(args):

   phi0, epsu, taus, tauf, delz, delh, delta, H, omega_rel = args

   


   # Set up meridional grid
   ngu = 800  # number of grid points for u and theta
   ngv = ngu+1  # number of grid points for v
   yu = np.linspace(-86,86,ngu)
   yu = np.radians(yu)
   dy = yu[1]-yu[0]
   yv = np.linspace(yu[0]-dy/2, yu[-1]+dy/2, ngv)


   dy = yu[1]-yu[0]    # m             grid spacing in y
   ngu = len(yu)
   ngv = ngu+1
   dyi = 1/dy       # 1/m           inverse grid spacing (multiplication clearer than division in some cases)
   y1 = np.radians(83)    # m             meridional location beyond which differential forcing vanishes

   # Parameters from Table 1 of SS09
   # name      value         units            description
   tavg = 310
   omega = 2*np.pi/86400 * omega_rel

   Rd = 287  # gas constant for dry air
   kappa = Rd / cp

   # Parameters in SS09 equations left undefined
   # name      value         units            description
   ps = 1000  # hPa           -- surface pressure (guess)
   ptop = 1000-H/100  # hPa           -- tropopause pressure (guess)

   # converting vertically averaged theta to temperature for pressure term
   # See Appendix C in Eusebi et al. (2026)
   C_theta = (ps**kappa - ptop**kappa) / (kappa * ps**kappa)

   # Parameters for time-stepping scheme (--twc guesses/empirical choices for stable performance)
   # name      value         units            description
   dt = 30  # s             -- model timestep
   tmax = 3e7  # s             -- end time of integration

   oshap = 4  #               -- Order of Shapiro filter to be used (valid options: none, 2, 4)
   nshap = 5  #               -- number of timesteps for Shapiro Filter smoothing
   nt = round(tmax/dt)  #  number of model timesteps
   iplot = 500  #               -- plotting interval (in timesteps)
   afilt = 0.05  #               -- Asselin filter coefficient
   fsmooth = 0.00  #               -- every iplot steps, set prognostic variables
   vdrag = 1e-5

   # increase drag in meridional momentum equation 
   # for high phi0 simulations for stability
   if phi0 > 19:
      vdrag = 2e-5

   # Define RCE theta profile (th_equil) based on Eusebi et al. (2026)
   phi = np.degrees(yu)
   th_equil = re_multi_plevel(phi, phi0, delh, tavg, pmin=ptop, pmax=ps)
   th_equil = np.mean(th_equil, axis=1)  # average over depth of troposphere

   # set to a flat profile for |y|>y1 using array logic
   iythmin = np.where(yu<-y1)[0]  # find indices corresponding to values of yu less than -y1
   iythmin = iythmin[-1]+1  # minimum yu-index for which RCE theta profile is valid
   iythmax = np.where(yu>y1)[0]  # find indices corresponding to values of yu greater than +y1
   iythmax = iythmax[0]-1  # maximum yu-index for which RCE theta profile is valid

   # if |yu|<=y1 then use the RCE profile,
   # else if yu<-y1: use the RCE profile at its southernmost valid point
   # else if yu>y1: use the RCE profile at its northernmost valid point
   th_equil = th_equil*(np.abs(yu)<=y1) + \
      th_equil[iythmin]*(yu<(-y1)) + \
      th_equil[iythmax]*(yu>y1)

   # Ensure equilibrium theta doesn't drop below stratosphere temp 200 K
   th_equil[th_equil<200] = 200

   # for latitudinally varying surface relaxation timescale, if tauf differs from taus
   tau_w = np.exp(-(yu-np.radians(phi0))**2/(2*np.radians(20)**2))  # weighting for conversion from taus to tauf
   tau = tau_w*taus + (1-tau_w)*tauf
   tau = tau*86400  # convert days to seconds

   # theta: initialize to isothermal initial state
   th1 = 0 * th_equil + 250
   th2 = th1  # th1, th2, th3 are values for use in leapfrogging/Asselin time scheme
   th3 = th1

   # zonal wind (u): initialize to zero everywhere
   u1 = np.zeros(ngu)
   u2 = u1
   u3 = u1

   # meridional wind (v): initialize to zero
   v1 = np.zeros(ngv)
   v2 = v1
   v3 = v1

   # time-averages for plotting and/or using to accelerate convergence to
   # steady state solutions
   ubar = 0*u1
   vbar = 0*v1
   thbar = 0*th1

   # Set up gross stability:
   delz = v2*0 + delz

   ### Integration in time
   it = 0
   while it < nt:
      it+=1
      if it%100000==0:
         print('progress', f'{np.round(it/nt*100,2)}%')
      # extended arrays for finite differencing
      uext = np.pad(u2, 1)  # assume u->0 at both ends
      thext = np.pad(th2, 1, 'edge')  # assume dtheta/dy is zero at both ends

      # upper layer mass divergence
      dvdy = dyi*((v2[1:]-v2[:-1]) - tan(yu)*(v2[1:]+v2[:-1])/2/Re)/Re

      # upper layer heat divergence
      dheatflux_dy = dyi*((v2[1:]*delz[1:]-v2[:-1]*delz[:-1]) - tan(yu)*(v2[1:]*delz[1:]+v2[:-1]*delz[:-1])/2/Re)/Re

      # CALCULATE THETA-TENDENCIES
      fthadv = delta/H * dheatflux_dy

      fthrad = (th_equil-th2)/tau

      # total theta-tendency is sum of components
      fth = -fthadv+fthrad

      # CALCULATE U-TENDENCIES
      fucorl = 2*omega*sin(yu)*(v2[:-1]+v2[1:])/2

      fuadvy = 1/Re*dyi*(v2[:-1]*((v2[:-1]+v2[1:])>0)*(uext[1:-1]-uext[:-2])+ \
                  (v2[1:]*((v2[:-1]+v2[1:])<=0)*(uext[2:]-uext[1:-1])))

      fuadvz = (dvdy>0)*u2*dvdy
      fumetric = tan(yu)/Re*u2*(v2[:-1]+v2[1:])/2
      frayd = epsu*u2


      # total u-tendency is sum of components
      fu = fucorl - fuadvy - fuadvz - frayd + fumetric

      # CALCULATE V-TENDENCIES
      fvcorl = 2*omega*sin(yv)*(uext[:-1]+uext[1:])/2

      fvpgrad = -1/Re * Rd * C_theta * dyi*(thext[1:]-thext[:-1])
      utempsq = (uext[:-1]**2+uext[1:]**2)/2
      fvmetric = tan(yv)/Re*utempsq
      fvfric = v2*vdrag

      # total v-tendency is sum of components
      fv = 0.5 * (-fvcorl + fvpgrad -fvmetric -fvfric)

      # ADVANCE QUANTITIES ONE STEP
      u3 = u1 + 2*dt*fu
      v3 = v1 + 2*dt*fv
      th3 = th1 + 2*dt*fth

      # APPLY SHAPIRO FILTER to eliminate gridscale noise every nshap timesteps
      # Use filter of order 'oshap'
      # See: Shapiro, 1975, "Linear Filtering," Mathematics of Computation,
      # Volume 29, #132, pp. 1094-1097
      # Table 2: Stencils for different-order filters:
      # filter order      x(i)        x(i+/-1)    x(i+/-2)    x(i+/-3)    x(i+/-4)
      #   1               2           1
      #   2               10          4           -1
      #   3               44          15          -6          1
      #   4               186         56          -28         8           -1
      if it%nshap == 0:
         if oshap==2:
            uext4 = np.pad(u3, 2)
            vext4 = np.pad(v3, 2)
            thex4 = np.pad(th3, 2, 'edge')
            u3 = 1/16*(-uext4[:-4]+4*uext4[1:-3]+10*uext4[2:-2]+4*uext4[3:-1]-uext4[4:])
            v3 = 1/16*(-vext4[:-4]+4*vext4[1:-3]+10*vext4[2:-2]+4*vext4[3:-1]-vext4[4:])
            th3 = 1/16*(-thex4[:-4]+4*thex4[1:-3]+10*thex4[2:-2]+4*thex4[3:-1]-thex4[4:])
         elif oshap==4:
            uext8 = np.pad(u3, 4)
            vext8 = np.pad(v3, 4)
            thex8 = np.pad(th3, 4, 'edge')
            u3 = 1/256*(186*u3+56*(uext8[3:-5]+uext8[5:-3])-28*(uext8[2:-6]+uext8[6:-2])+8*(uext8[1:-7]+uext8[7:-1])-1*(uext8[:-8]+uext8[8:]))
            v3 = 1/256*(186*v3+56*(vext8[3:-5]+vext8[5:-3])-28*(vext8[2:-6]+vext8[6:-2])+8*(vext8[1:-7]+vext8[7:-1])-1*(vext8[:-8]+vext8[8:]))
            th3 = 1/256*(186*th3+56*(thex8[3:-5]+thex8[5:-3])-28*(thex8[2:-6]+thex8[6:-2])+8*(thex8[1:-7]+thex8[7:-1])-1*(thex8[:-8]+thex8[8:]))

      # APPLY ASSELIN TIME-FILTER
      u1 = u2+afilt*(u1-2*u2+u3)
      v1 = v2+afilt*(v1-2*v2+v3)
      th1 = th2+afilt*(th1-2*th2+th3)

      u2 = u3
      v2 = v3
      th2 = th3

      # accumulate average values over iplot timesteps
      ubar = ubar+1/iplot*u2
      vbar = vbar+1/iplot*v2
      thbar = thbar+1/iplot*th2

      if it%iplot==0:
         # every iplot timesteps, smooth prognostic variables towards their
         # mean values over the past averaging period
         th1 = (1-fsmooth)*th1+fsmooth*thbar
         th2 = th1
         th3 = th1
         
         u1 = (1-fsmooth)*u1+fsmooth*ubar
         u2 = u1
         u3 = u1
         
         v1 = (1-fsmooth)*v1+fsmooth*vbar
         v2 = v1
         v3 = v1

         # reset average values to zero
         thbar = 0*thbar
         ubar = 0*ubar
         vbar = 0*vbar

         # if the model goes unstable, decrease the timestep and try again
         if (np.sum(np.isnan(u2)*1)>0):
            dt = dt/1.5
            nt = round(tmax/dt)
            it = 0

            print('WARNING: NaNs found in zonal wind model probably unstable')
            print(f'Decreasing timestep by factor of 1.5, to {dt} s')
            print('Will attempt to re-initialize with th=th_equil, u=v=0')

            # reset prognostic variables so condition doesn't immediately activate again
            th1 = 0*th_equil+250
            th2 = th1
            th3 = th1
            thbar = 0*th1

            u1 = np.zeros(ngu)
            u2 = u1
            u3 = u2
            ubar = 0*u1

            v1 = np.zeros(ngv)
            v2 = v1
            v3 = v2
            vbar = 0*v1

   
   # Create xarray dataset with final results (on meridional grid with points yu)
   ds = xr.Dataset({
       'u': (['lat'], u3), 
       'theta': (['lat'], th3)
   }, coords={'lat': np.degrees(yu), 'phi0': phi0})
   
   # Create v DataArray on staggered grid (yv) and interpolate to regular grid (yu)
   v = xr.DataArray(v3, dims=['lat_v'], coords={'lat_v': np.degrees(yv)}).interp(lat_v=ds.lat)
   ds['v'] = v

   # Return dataset of results
   return ds


def axisymm_hadley_1p5_layer(epsu, taus, tauf, delz, delh, delta, H, omega_rel):
   """
   Run the one-layer Hadley model for varying seasonality phi0.

   Notes
   - Set taus and tauf to equal values for constant relaxation timescale.
     Otherwise, set taus to a lower value for latitudinal structure in Appendix
     A of Eusebi et al. (2026). 
   - Set omega_rel to 1 for Earth's rotation rate.
   - Radiative equilibrium (RE) profiles are calculated using the function in 
     re_profiles.py, which are described in Eusebi et al. (2026).  
   - Set epsu to 1e-9 for inviscid limit (drag minimally affecting flow)
   - If desired, gross stability delz can be modified in this function 
     to be inputted as an array of the same size as yv to add latitudinal structure.

   Args:
       epsu (float): Rayleigh drag coefficient [1/s]; damps zonal wind.
       taus (float): Newtonian relaxation timescale near phi0 [days];
           radiative relaxation toward RE theta.
       tauf (float): Newtonian relaxation timescale far from phi0 [days];
           radiative relaxation toward RE theta.
       delz (float): Gross stability in temperature units [K];
       delh (float): Meridional potential temperature contrast [K]; controls
           RE equator-to-pole gradient.
       delta (float): Layer thickness [m]; used for mass and heat transport.
       H (float): Tropopause thickness [Pa]
       omega_rel (float): Earth rotation rate multiplier; actual rate is
           omega * (2*pi/86400) rad/s (1 = Earth value).
   """

   ### Set phi0 array to calculate Hadley circulations for
   phi0s = np.array([0, 20])

   ### Loop over phi0 values
   # done with multiprocessing to speed up calculation
   args = ((phi0, epsu, taus, tauf, delz, delh, delta, H, omega_rel) for phi0 in phi0s)
   with Pool(processes=len(phi0s)) as pool:
      # Map the function over the range of tasks
      results = pool.map(run_hadley, args)
      
   ds = xr.concat(results, dim='phi0')

   # Get streamfunction in units kg / s
   psi = -ds['v'] * 2 * np.pi * Re * np.cos(np.radians(ds.lat)) * delta / g

   # Get heat transport by circulation in units W
   heatflux = -psi * delz * cp

   ds['psi'] = psi
   ds['heatflux'] = heatflux

   # Add code to save data to netcdf file or plot figuresif desired
   ds.to_netcdf("/resnick/groups/esm/reusebi/HadleyLayerModels/test.nc")
   return ds

# Example run script:
# python axisymm_hadley_1p5_layer.py --epsu 1e-9 --taus 50 --tauf 50 --delz 30 --delh 120 --delta 20000 --H 80000 --omega_rel 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input file and save to output directory.")
    parser.add_argument("--epsu", type=float, required=True, help="Rayleigh drag coefficient in 1/s")
    parser.add_argument("--taus", type=float, required=True, help="Newtonian relaxation timescale in days")
    parser.add_argument("--tauf", type=float, required=True, help="Newtonian relaxation timescale in days")
    parser.add_argument("--delz", type=float, required=True, help="Potential temperature stability in K")
    parser.add_argument("--delh", type=float, required=True, help="Meridional pot temperature difference in K")
    parser.add_argument("--delta", type=float, required=True, help="Thickness of layers in Pa")
    parser.add_argument("--H", type=float, required=True, help="Tropopause height in Pa")
    parser.add_argument("--omega_rel", type=float, required=True, help="Earth rotation rate multiplier")

    args = parser.parse_args()
    axisymm_hadley_1p5_layer(args.epsu, args.taus, args.tauf, args.delz, args.delh, args.delta, args.H, args.omega_rel)