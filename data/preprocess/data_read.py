import os
import numpy as np
import xarray
import matplotlib.pyplot as plt
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

# file = os.path.join(f'{hp.reanalysis_dataset_dir}/meta-data/', 'HadISST_sst.nc')
# data = xarray.open_dataset(file, cache=True, decode_times=True)['sst']
# print(data)
# la = data.coords["latitude"]
# lc = data.coords["longitude"]
# data = data.loc[dict(longitude=lc[(lc >= 120) & (lc <= 280)], latitude=la[(la >= -30) & (la <= 30)])]
# sample = data.sel(time='2015-12-16')
# print(min(sample.values))
# sample.plot()
# plt.show()

uwind = np.load(f"{hp.reanalysis_npz_dir}/{'uwind-resolve'}.npz")['uwind']
vwind = np.load(f"{hp.reanalysis_npz_dir}/{'vwind-resolve'}.npz")['vwind']
sst = np.load(f"{hp.reanalysis_npz_dir}/{'sst-resolve'}.npz")['sst']
sshg = np.load(f"{hp.reanalysis_npz_dir}/{'sshg'}.npz")['sshg']
thflx = np.load(f"{hp.reanalysis_npz_dir}/{'thflx'}.npz")['thflx']

print(uwind.shape, vwind.shape, sst.shape, sshg.shape, thflx.shape)

# observe sst
# Dimensions:    (lat: 180, lon: 360, nbnds: 2, time: 470)
# Coordinates:
#   * lat        (lat) float32 89.5 88.5 87.5 86.5 ... -86.5 -87.5 -88.5 -89.5
#   * lon        (lon) float32 0.5 1.5 2.5 3.5 4.5 ... 356.5 357.5 358.5 359.5
#   * time       (time) datetime64[ns] 1981-12-01 1982-01-01 ... 2021-01-01
# Dimensions without coordinates: nbnds
# Data variables:
#     sst        (time, lat, lon) float32 ...
#     time_bnds  (time, nbnds) datetime64[ns] ...


# observe wind
# Dimensions:    (latitude: 241, longitude: 480, time: 476)
# Coordinates:
#   * longitude  (longitude) float32 0.0 0.75 1.5 2.25 ... 357.75 358.5 359.25
#   * latitude   (latitude) float32 90.0 89.25 88.5 87.75 ... -88.5 -89.25 -90.0
#   * time       (time) datetime64[ns] 1980-01-01 1980-02-01 ... 2019-08-01
# Data variables:
#     u10        (time, latitude, longitude) float32 ...


# observe sshg
# Dimensions:  (lat: 418, lon: 360, time: 492)
# Coordinates:
#   * lon      (lon) float32 0.5 1.5 2.5 3.5 4.5 ... 355.5 356.5 357.5 358.5 359.5
#   * lat      (lat) float32 -74.5 -74.16667 -73.83334 ... 64.16566 64.499
#   * time     (time) datetime64[ns] 1980-01-01 1980-02-01 ... 2020-12-01
# Data variables:
#     sshg     (time, lat, lon) float32 ...


# observe thflx
# Dimensions:  (lat: 418, lon: 360, time: 492)
# Coordinates:
#   * lon      (lon) float32 0.5 1.5 2.5 3.5 4.5 ... 355.5 356.5 357.5 358.5 359.5
#   * lat      (lat) float32 -74.5 -74.16667 -73.83334 ... 64.16566 64.499
#   * time     (time) datetime64[ns] 1980-01-01 1980-02-01 ... 2020-12-01
# Data variables:
#     thflx    (time, lat, lon) float32 ...


# reanalysis sst
# Dimensions:    (latitude: 180, longitude: 360, nv: 2, time: 1811)
# Coordinates:
#   * time       (time) datetime64[ns] 1870-01-16T11:59:59.505615234 ... 2020-11-16T12:00:00
#   * latitude   (latitude) float32 89.5 88.5 87.5 86.5 ... -87.5 -88.5 -89.5
#   * longitude  (longitude) float32 -179.5 -178.5 -177.5 ... 177.5 178.5 179.5
# Dimensions without coordinates: nv
# Data variables:
#     time_bnds  (time, nv) float32 ...
#     sst        (time, latitude, longitude) float32 ...


# reanalysis sshg
# Dimensions:  (lat: 180, lon: 360, time: 1980)
# Coordinates:
#   * lat      (lat) float64 -89.5 -88.5 -87.5 -86.5 -85.5 ... 86.5 87.5 88.5 89.5
#   * lon      (lon) float64 0.5 1.5 2.5 3.5 4.5 ... 355.5 356.5 357.5 358.5 359.5
#   * time     (time) object 1850-01-16 12:00:00 ... 2014-12-16 12:00:00
# Data variables:
#     zos      (time, lat, lon) float32 ...


# reanalysis thflx
# Dimensions:  (lat: 180, lon: 360, time: 1980)
# Coordinates:
#   * lat      (lat) float64 -89.5 -88.5 -87.5 -86.5 -85.5 ... 86.5 87.5 88.5 89.5
#   * lon      (lon) float64 0.5 1.5 2.5 3.5 4.5 ... 355.5 356.5 357.5 358.5 359.5
#   * time     (time) object 1850-01-16 12:00:00 ... 2014-12-16 12:00:00
# Data variables:
#     hfds     (time, lat, lon) float32 ...

