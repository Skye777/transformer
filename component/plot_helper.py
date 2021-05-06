"""
@author: Skye Cui
@file: plot_helper.py
@time: 2021/5/4 10:14
@description: 
"""
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np


def grid_position(shape, extent):
    """Return `lons`, `lats`"""
    ntimes, nlats, nlons = shape
    latmin, latmax, lonmin, lonmax = extent
    lats = np.linspace(latmin, latmax, nlats)
    lons = np.linspace(lonmin, lonmax, nlons)
    lons, lats = np.meshgrid(lons, lats)
    print(lons, lats)

    return lons, lats


def plot_helper(data):
    projection = ccrs.PlateCarree(central_longitude=180)
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    lonmin, lonmax = -60, 100
    latmin, latmax = -40, 40
    extent = [latmin, latmax, lonmin, lonmax]
    lons, lats = grid_position(shape=data.shape, extent=extent)

    fig = plt.figure()
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 2),
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode

    for i, ax in enumerate(axgr):
        ax.coastlines()
        ax.set_xticks(np.linspace(lonmin, lonmax, 5), crs=projection)
        ax.set_yticks(np.linspace(latmin, latmax, 5), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        p = ax.contourf(lons, lats, data[i, ...],
                        transform=projection,
                        cmap='RdYlBu_r')

    axgr.cbar_axes[0].colorbar(p)

    plt.show()


if __name__ == '__main__':
    plot_helper(np.random.random((6, 160, 320)))