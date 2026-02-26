"""
Radiative Equilibrium (RE) Temperature Profiles

The functions in this module compute the RE potential temperature profile as a function of latitude
and pressure level according to Eqs. (2-4) of Eusebi et al. (2026). The profiles are used 
to study Hadley circulation and tropical atmospheric dynamics. The profiles are calculated to have
a constant globally averaged RE surface temperature specified by tavg.

The surface temperature profile follow the form:
T = tavg - dh*(sin²(φ) - 2*sin(φ0)*sin(φ) - 1/3)

References:
- Eusebi et al. (2026)
"""

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt


def re_single_plevel(phi, phi0=0, dh=120, tavg=310, p=1000):
    """
    Calculate RE potential temperature profile at a single pressure level.
    
    Parameters
    ----------
    phi : array_like
        Latitude coordinates in degrees. Can be scalar or array. Must be evenly
        spaced and span (or nearly span) -90 to 90 degrees.
    phi0 : float, optional
        Reference latitude in degrees where maximum heating occurs.
        Typically represents the latitude of maximum solar heating. Default is 0.
    dh : float, optional
        Temperature contrast parameter in Kelvin. Controls the amplitude
        of the latitudinal temperature variation. Default is 120.
    tavg : float, optional
        Average global RE surface temperature in Kelvin. Default is 310.
    p : float, optional
        Pressure level in hPa at which to calculate the temperature profile. 
        Default is 1000.
    
    Returns
    -------
    numpy.ndarray
        Temperature in Kelvin at the specified pressure level and latitudes.
        Same shape as input phi array.
    """
    # Convert latitude coordinates from degrees to radians
    phi = np.radians(phi)
    phi0 = np.radians(phi0)
    
    
    p0 = 1000 # Reference pressure level (hPa) for scaling
    t_strat = 200 # Stratospheric temperature in kelvin

    # Calculate basic temperature profile using Bordoni formulation
    t_surf = tavg - dh*(sin(phi)**2 - 2*sin(phi0)*sin(phi) - 1/3)

    # Apply minimum temperature constraint (200K floor)
    t_surf[t_surf < t_strat] = t_strat
    
    # Normalize temperature relative to 200K baseline for energy balance calculation
    t_surf = t_surf - t_strat
    tavg -= t_strat
    
    # Calculate area-weighted mean temperature to enforce energy balance
    temp_tot = np.sum(t_surf * cos(phi)) / np.sum(cos(phi))
    dif = temp_tot - tavg
    
    # Apply cooling factor to maintain specified average temperature
    coolfactor = 1 - dif * np.mean(cos(phi)) / np.mean(t_surf * cos(phi))
    t_surf = t_surf * coolfactor

    # Restore 200K baseline
    t_surf += t_strat

    # Get temperature at vertical level p using surface T and vertical structure formula
    temp_p = t_strat * (1 + ((t_surf/t_strat)**4 - 1) * (p/p0)**(3.5))**(1/4) 
    theta_p = temp_p * (p0/p)**(2/7) # convert to potential temperature

    return theta_p

def re_multi_plevel(phi, phi0, dh, tavg, pmin=200, pmax=1000):
    """
    Calculate 2D RE potential temperature field (latitude × pressure).
    
    Parameters
    ----------
    phi : array_like
        Latitude coordinates in degrees. Can be scalar or array.
    phi0 : float
        Reference latitude in degrees where maximum heating occurs.
        Typically represents the latitude of maximum solar heating.
    dh : float
        Temperature contrast parameter in Kelvin. Controls the amplitude
        of the latitudinal temperature variation at the surface.
    tavg : float
        Average temperature in Kelvin at 1000 hPa reference level.
        Represents the mean tropical surface temperature.
    pmin : float, optional
        Minimum pressure level in hPa (default: 200). Top of the 
        atmospheric column for the calculation.
    pmax : float, optional
        Maximum pressure level in hPa (default: 1000). Bottom of the
        atmospheric column, typically near surface.
    
    Returns
    -------
    numpy.ndarray
        Temperature field in Kelvin with shape (len(phi), 1000).
        First dimension corresponds to latitude, second to pressure levels
        from pmin to pmax with 1000 equally-spaced points.
    
    The resulting field can be used as initial conditions or forcing for
    atmospheric circulation models.
    """
    # Convert latitude coordinates from degrees to radians
    phi = np.radians(phi)
    phi0 = np.radians(phi0)

    # Stratospheric temperature in kelvin
    t_strat = 200
    
    # Reference pressure level (hPa) for scaling
    p0 = 1000
    
    # Calculate basic temperature profile using Bordoni formulation
    t_surf = tavg - dh*(sin(phi)**2 - 2*sin(phi0)*sin(phi) - 1/3)

    # Apply minimum temperature constraint (200K floor)
    t_surf[t_surf < t_strat] = t_strat
    
    # Normalize temperature relative to 200K baseline for energy balance calculation
    t_surf = t_surf - t_strat
    tavg -= t_strat
    
    # Calculate area-weighted mean temperature to enforce energy balance
    temp_tot = np.sum(t_surf * cos(phi)) / np.sum(cos(phi))
    dif = temp_tot - tavg
    
    # Apply cooling factor to maintain specified average temperature
    coolfactor = 1 - dif * np.mean(cos(phi)) / np.mean(t_surf * cos(phi))
    t_surf = t_surf * coolfactor
    
    # Restore 200K baseline
    t_surf += t_strat

    # Create pressure coordinate array with specified range and resolution
    p = np.linspace(pmin, pmax, 1000)
    
    # Create 2D meshgrid for latitude-pressure temperature field
    # indexing='ij' ensures proper matrix orientation (lat × pressure)
    TEMP, P = np.meshgrid(t_surf, p, indexing='ij')

    # Calculate 2D temperature field using vertical structure formula
    temp_p = t_strat * (1 + ((TEMP/t_strat)**4 - 1) * (P/p0)**(3.5))**(1/4) 
    theta_p = temp_p * (p0/P)**(2.0/7)
    
    return theta_p

