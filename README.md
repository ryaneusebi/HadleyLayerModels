# HadleyLayerModels

This github repository contains code for several simple idealized models for the Hadley circulation, including some numerical and some semi-analytical models. The models in this repository as of Mar. 3rd, 2026 are all described in Eusebi and Schneider (2026). 

The models featured are listed here, along with the filename associated with the code for each model, and described further in the subheaders below:
- A 1.5-layer axisymmetric model of the Hadley cell (axisymm_hadley_1p5_layer.py)
- A 2-layer model of the Hadley cell (axisymm_hadley_2_layer.py)

The files generate outputs in netcdf files using xarray. The netcdf file output has two coordinates: 'lat' and 'phi0'. 'phi0' refers to the latitude of maximum insolation in the radiative equilibrium profile (see Eusebi and Schneider (2026)) and 'lat' refers to the different latitude points within a given simulation and phi0. In each file, the user can set the phi0s they want to calculate Hadley circulation profiles for. By default, the code uses multiprocessing to calculate the profiles for all of these phi0 values at once - the user can change this if they do not wish to use multiprocessing. The user can specify different planetary climate variables, such as rotation rate, via the inputs as discussed in the files.

The files are written so they can be run directly using the example commands (with example, default input arguments) listed above the if __name__ == '__main__' block; however, the user can create another script to automate parameter sweeps for calculating various circulations if they wish. 

See specific documentation for each of the files below. If any questions come up or you are using this code for projects, feel free to reach out to the author at reusebi@caltech.edu if you have any questions.

## 1.5-layer axisymmetric model (LM1.5)
