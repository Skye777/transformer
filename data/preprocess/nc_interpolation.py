import os
import numpy as np
import matplotlib.pyplot as plt
import xarray
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

dic = [
    [f'{hp.observe_dataset_dir}/meta-data/sst.mnmean.nc', f'{hp.observe_dataset_dir}/interp-data/sst.nc', 'sst', 'lat'],
    [f'{hp.observe_dataset_dir}/meta-data/sshg/sshg.mon.mean1980-2020.nc', f'{hp.observe_dataset_dir}/interp-data/sshg.nc', 'sshg', 'lat'],
    [f'{hp.observe_dataset_dir}/meta-data/thflx/thflx.mon.mean1980-2020.nc', f'{hp.observe_dataset_dir}/interp-data/thflx.nc', 'thflx', 'lat'],
    [f'{hp.observe_dataset_dir}/meta-data/wind/uwind.mon.mean1980-2019.nc', f'{hp.observe_dataset_dir}/interp-data/uwind.nc', 'u10', 'latitude'],
    [f'{hp.observe_dataset_dir}/meta-data/wind/vwind.mon.mean1980-2019.nc', f'{hp.observe_dataset_dir}/interp-data/vwind.nc', 'v10', 'latitude'],
    [f'{hp.reanalysis_dataset_dir}/meta-data/sshg/sshg.mon.mean1850-2014.nc', f'{hp.reanalysis_dataset_dir}/interp-data/sshg.nc', 'zos', 'lat'],
    [f'{hp.reanalysis_dataset_dir}/meta-data/thflx/thflx.mon.mean1850-2014.nc', f'{hp.reanalysis_dataset_dir}/interp-data/thflx.nc', 'hfds', 'lat']
]


def interp(attr):
    basefile, savefile, var, coords = attr
    dataset = xarray.open_dataset(basefile, cache=True, decode_times=True)
    data = dataset[var]

    lat_len, lon_len = 160, 320
    lon = np.linspace(120, 280, lon_len)
    lat = np.linspace(-39.5, 40, lat_len)

    if coords == 'lat':
        dense_data = data.interp(lat=lat, lon=lon)
    else:
        dense_data = data.interp(latitude=lat, longitude=lon)
    print(dense_data)
    print(dense_data.shape)
    dense_data.to_netcdf(savefile)


def main():
    for i in dic:
        interp(i)


if __name__ == "__main__":
    main()


# print(sst)
# Coordinates:
#   * lat      (lat) float32 89.5 88.5 87.5 86.5 85.5 ... -86.5 -87.5 -88.5 -89.5
#   * lon      (lon) float32 0.5 1.5 2.5 3.5 4.5 ... 355.5 356.5 357.5 358.5 359.5
#   * time     (time) datetime64[ns] 1850-01-01 1850-02-01 ... 2017-12-01
# la = var.coords["lat"]
# lc = var.coords["lon"]
# var = var.loc[dict(lon=lc[(lc >= 120) & (lc <= 280)], lat=la[(la >= -40) & (la <= 40)])]
# sample = var.sel(time='2015-12-01')
# sample.plot()
# plt.show()
# print(var)
# basefile, savefile, var = dic[5]
# dataset = xarray.open_dataset(basefile, cache=True, decode_times=True)
# data = dataset[var]
# la = data.coords["lat"]
# lc = data.coords["lon"]
# data = data.loc[dict(lon=lc[(lc >= 120) & (lc <= 280)], lat=la[(la >= -40) & (la <= 40)])]
# print(data)
# sample = data.sel(time='2014-12-16')
# sample.plot()
# plt.show()
#
#
# lat_len, lon_len = 160, 320
# lon = np.linspace(120, 280, lon_len)
# lat = np.linspace(-39.5, 40, lat_len)
# dense_data = data.interp(lat=lat, lon=lon)
# sample = dense_data.sel(time='2014-12-16')
# sample.plot()
# plt.show()
# print(dense_data)
# print(dense_data.shape)


# # print(uwnd)
# # Coordinates:
# #   * lat      (lat) float32 90.0 88.0 86.0 84.0 82.0 ... -84.0 -86.0 -88.0 -90.0
# #   * lon      (lon) float32 0.0 2.0 4.0 6.0 8.0 ... 350.0 352.0 354.0 356.0 358.0
# #   * time     (time) datetime64[ns] 1851-01-01 1851-02-01 ... 2014-12-0
# # select the useful time
# var = var.loc['1982-01-01':, :, :]
# la = var.coords["lat"]
# lc = var.coords["lon"]
# var = var.loc[dict(lon=lc[(lc >= 120) & (lc <= 280)], lat=la[(la >= -30) & (la <= 30)])]
# # print(var)
# # Coordinates:
# #   * lat      (lat) float32 30.0 28.0 26.0 24.0 22.0 ... -24.0 -26.0 -28.0 -30.0
# #   * lon      (lon) float32 120.0 122.0 124.0 126.0 ... 274.0 276.0 278.0 280.0
# #   * time     (time) datetime64[ns] 1982-01-01 1982-02-01 ... 2014-12-01
#
# sample = var.sel(time='1982-12-01')
# sample.plot()
# plt.show()
#
# lat_len, lon_len = sample.shape
# print("origin shape: ", sample.shape)
# lon = np.linspace(120.0, 280.0, (lon_len-1) * 8)
# lat = np.linspace(-30.0, 30.0, (lat_len-1) * 8)
# dense_var = var.interp(lat=lat, lon=lon)
# dense_sample = dense_var.sel(time='1982-12-01')
# print(dense_var)
# print("new shape: ", dense_var.shape)
# dense_sample.plot()
# plt.show()
# dense_var.to_netcdf(savefile)


# # print(sshg)
# # Coordinates:
# #   * lon      (lon) float32 0.5 1.5 2.5 3.5 4.5 ... 355.5 356.5 357.5 358.5 359.5
# #   * lat      (lat) float32 -74.5 -74.16667 -73.83334 ... 64.16566 64.499
# #   * time     (time) datetime64[ns] 1980-01-01 1980-02-01 ... 1980-12-01
# la = var.coords["lat"]
# lc = var.coords["lon"]
# var = var.loc[dict(lon=lc[(lc >= 120) & (lc <= 280)], lat=la[(la >= -30) & (la <= 30)])]
# # print(var)
# # Coordinates:
# #   * lon      (lon) float32 120.5 121.5 122.5 123.5 ... 276.5 277.5 278.5 279.5
# #   * lat      (lat) float32 -29.83365 -29.50032 -29.16699 ... 29.49925 29.83258
# #   * time     (time) datetime64[ns] 1982-01-01 1982-02-01 ... 2014-12-01
# sample = var.sel(time='1982-12-01')
# sample.plot()
# plt.show()
#
# lat_len, lon_len = sample.shape
# lon = np.linspace(120.5, 279.5, lon_len)
# lat = np.linspace(-29.5, 29.5, int(lat_len / 3))
# dense_sshg = var.interp(lat=lat, lon=lon)
# month12 = dense_sshg.sel(time='1982-12-01')
# month12.plot()
# print(dense_sshg)
# print(dense_sshg.shape)
# plt.show()
# dense_sshg.to_netcdf(savefile)
