import os
import numpy as np
import netCDF4 as nc
import xarray

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

sst = xarray.open_dataset(f'{hp.observe_dataset_dir}/interp-data/sst.nc')['sst'].loc['1981-12-01':'2019-08-01', :, :]
sshg = xarray.open_dataset(f'{hp.observe_dataset_dir}/interp-data/sshg.nc')['sshg'].loc['1981-12-01':'2019-08-01', :, :]
thflx = xarray.open_dataset(f'{hp.observe_dataset_dir}/interp-data/thflx.nc')['thflx'].loc['1981-12-01':'2019-08-01', :, :]
uwind = xarray.open_dataset(f'{hp.observe_dataset_dir}/interp-data/uwind.nc')['u10'].loc['1981-12-01':'2019-08-01', :, :]
vwind = xarray.open_dataset(f'{hp.observe_dataset_dir}/interp-data/vwind.nc')['v10'].loc['1981-12-01':'2019-08-01', :, :]
zos = xarray.open_dataset(f'{hp.reanalysis_dataset_dir}/interp-data/sshg.nc')['zos'].loc['1870-01-16':'2014-12-16', :, :]
hfds = xarray.open_dataset(f'{hp.reanalysis_dataset_dir}/interp-data/thflx.nc')['hfds'].loc['1870-01-16':'2014-12-16', :, :]

sst = np.array(sst)
sshg = sshg.fillna(0).values
thflx = thflx.fillna(0).values
uwind = np.array(uwind)
vwind = np.array(vwind)
zos = zos.fillna(0).values
hfds = hfds.fillna(0).values

print(sst.shape)
print(sshg.shape)
print(thflx.shape)
print(uwind.shape)
print(vwind.shape)
print(zos.shape)
print(hfds.shape)

data = {'sst': sst}
np.savez(f'{hp.observe_npz_dir}/sst.npz', **data)

data = {'sshg': sshg}
np.savez(f'{hp.observe_npz_dir}/sshg.npz', **data)

data = {'thflx': thflx}
np.savez(f'{hp.observe_npz_dir}/thflx.npz', **data)

data = {'uwind': uwind}
np.savez(f'{hp.observe_npz_dir}/uwind.npz', **data)

data = {'vwind': vwind}
np.savez(f'{hp.observe_npz_dir}/vwind.npz', **data)

data = {'sshg': zos}
np.savez(f'{hp.reanalysis_npz_dir}/sshg.npz', **data)

data = {'thflx': hfds}
np.savez(f'{hp.reanalysis_npz_dir}/thflx.npz', **data)

# uwind = []
# vwind = []
# for f in os.listdir(base_path):
#     if os.path.isfile(f'{base_path}/{f}'):
#         uwind.append(nc.Dataset(f'{base_path}/{f}').variables['uwnd'][0])
#         vwind.append(nc.Dataset(f'{base_path}/{f}').variables['vwnd'][0])
#
# for m in os.listdir(f'{base_path}/Y2019'):
#     temp_u = []
#     temp_v = []
#     print(m)
#     for f in os.listdir(f'{base_path}/Y2019/{m}'):
#         temp_u.append(np.mean(nc.Dataset(f'{base_path}/Y2019/{m}/{f}').variables['uwnd'], axis=0))
#         temp_v.append(np.mean(nc.Dataset(f'{base_path}/Y2019/{m}/{f}').variables['vwnd'], axis=0))
#     temp_u = np.array(temp_u)
#     temp_v = np.array(temp_v)
#     uwind.append(np.mean(temp_u, axis=0))
#     vwind.append(np.mean(temp_v, axis=0))
#
# for m in os.listdir(f'{base_path}/Y2020'):
#     temp_u = []
#     temp_v = []
#     print(m)
#     for f in os.listdir(f'{base_path}/Y2020/{m}'):
#         temp_u.append(np.mean(nc.Dataset(f'{base_path}/Y2020/{m}/{f}').variables['uwnd'], axis=0))
#         temp_v.append(np.mean(nc.Dataset(f'{base_path}/Y2020/{m}/{f}').variables['vwnd'], axis=0))
#     temp_u = np.array(temp_u)
#     temp_v = np.array(temp_v)
#     uwind.append(np.mean(temp_u, axis=0))
#     vwind.append(np.mean(temp_v, axis=0))


