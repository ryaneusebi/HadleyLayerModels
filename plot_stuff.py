import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset('/resnick/groups/esm/reusebi/HadleyLayerModels/test.nc')

ds = ds.sel(phi0=20)
plt.plot(ds.lat, ds.u)
plt.savefig('u.png')
plt.close()

plt.plot(ds.lat, ds.theta)
plt.savefig('theta.png')
plt.close()

plt.plot(ds.lat, ds.psi)
plt.savefig('psi.png')
plt.close()

plt.plot(ds.lat, ds.heatflux)
plt.savefig('heatflux.png')
plt.close()
