import openeo
from openeo.processes import ProcessBuilder
import os
import pyrosm
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geowombat as gw
import pickle
from skimage import exposure
import numpy as np

from params import *

# get Open Street Map file
def get_openstreetmap(city, osm_path=OSM_PATH, coordinates=None):
    # create path if not there
    if not os.path.isdir(osm_path):
        os.makedirs(osm_path)
    # download OSM map
    osm_map = pyrosm.get_data(city, directory=osm_path)

# read building geometry out of OSM
def get_buildings(city, osm_path=OSM_PATH, coord_bounds=None):
    osm_city_path = os.path.join(osm_path, f'{city}.osm.pbf')
    if coord_bounds is None:
        osm_map = pyrosm.OSM(osm_city_path)
    else:
        osm_map = pyrosm.OSM(osm_city_path, coord_bounds)
    building_map = osm_map.get_buildings()
    print(f'Building map for {city} created.')
    coord_bounds = building_map.geometry.total_bounds

    return coord_bounds, building_map

# get satellite images as GTiff
def get_sat_img(coord_bounds, city, image_path=IMAGE_DATA_PATH):
    
    # create sat_image path if not there:
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    
    # get connection
    connection = openeo.connect('https://openeocloud.vito.be/openeo/1.0.0')
    # authentication
    #connection.authenticate_basic("username", "password")
    connection.authenticate_oidc()

    sat_data = connection.load_collection(
        'SENTINEL2_L2A',
        spatial_extent={"west": coord_bounds[0], "south": coord_bounds[1], "east": coord_bounds[2], "north": coord_bounds[3]},
        temporal_extent=["2023-03-01", "2023-07-01"],
        max_cloud_cover=30,
        bands=['B02', 'B03', 'B04', 'B08'] # B, G, R, infrared
    )

    # get mean of bands
    sat_data_mean = sat_data.mean_time()

    # save sat_image data
    sat_result = sat_data_mean.save_result(format='GTiff')
    file_name = os.path.join(image_path, f'{city}.tif')
    job = sat_result.create_job()
    job.start_and_wait()
    job.get_results().download_file(file_name)

def rasterize_buildings(map, building_map):
    building_raster = rasterio.features.rasterize(
        shapes=building_map.geometry, 
        out_shape=(map.shape[1], map.shape[2]),
        transform=map.transform, 
        fill=0,
        all_touched=True, 
        dtype=rasterio.uint8
    )
    return building_raster

def plot_image_data(image_data):
    fig = plt.figure(figsize=(16,10))
    ax_list = fig.subplots(2,4)

    # plot overlapping buildings
    image_data['Buildings'].plot(ax=ax_list[0][0], zorder=2)
    image_data['R'].plot(ax=ax_list[0][0], cmap='gray', zorder=1, add_colorbar=False)

    # plot buildings
    image_data['Buildings'].plot(ax=ax_list[0][1])

    # plot RGB
    image_data['RGB'].gw.imshow(robust=True, flip=False, ax=ax_list[0][2])

    # plot NIRGB
    image_data['NIRGB'].gw.imshow(robust=True, flip=False, ax=ax_list[0][3])

    # plot singe channels
    image_data['R'].plot(ax=ax_list[1][0], cmap='gray', add_colorbar=False)
    image_data['G'].plot(ax=ax_list[1][1], cmap='gray', add_colorbar=False)
    image_data['B'].plot(ax=ax_list[1][2], cmap='gray', add_colorbar=False)
    image_data['NIR'].plot(ax=ax_list[1][3], cmap='gray', add_colorbar=False)

    # adjust plot
    for title, ax in zip(image_data.keys(),ax_list.flatten()):
        ax.set_title(title)
        ax.title.set_fontsize(14)
        ax.axis('off')
        ax.set_aspect('equal')  
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()  

def save_data(image_data, city, path=IMAGE_DATA_PATH):
    # save dictionar as pkl
    file_name=f'{city}.pkl'

    with open(os.path.join(path, file_name), 'wb') as file:
        pickle.dump(image_data, file)
    print(f'Image dataset {city} written.')

def color_correct(channels,r_cor =1, g_cor=1, b_cor=1):
    return channels[0]*r_cor, channels[1]*g_cor, channels[2]*b_cor

def equalize(channel):
    values = channel.values.copy()
    # replace nan values by 0
    values = np.nan_to_num(values)
    return exposure.equalize_hist(values)

def img_process(city, building_map, path=IMAGE_DATA_PATH):
    
    file_path = os.path.join(path, f'{city}.tif')

    with gw.config.update(sensor='bgrn'):
        with gw.open(file_path) as sat_image:
            # match sat_image CRS
            building_map = building_map.to_crs(sat_image.crs)

            # create rgb
            rgb = sat_image.where(sat_image != 0).sel(band=["red", "green", "blue"])
            # histogram equalization: rgb.values gives the np.array
            rgb.values = equalize(rgb)
            # manually correct color values
            #rgb.values[0],rgb.values[1], rgb.values[2] = color_correct(rgb.values)
            
            # create nirgb
            nirgb = sat_image.where(sat_image != 0).sel(band=["nir", "green", "blue"])
            # histogram equalization
            nirgb.values = equalize(nirgb)
            
            # r
            r = sat_image.where(sat_image != 0).sel(band=['red'])
            # histogram equalization
            r.values = equalize(r)

            # g
            g = sat_image.where(sat_image != 0).sel(band=['green'])
            # histogram equalization
            g.values = equalize(g)

            # b
            b = sat_image.where(sat_image != 0).sel(band=['blue'])
            # histogram equalization
            b.values = equalize(b)

            # nir
            nir = sat_image.where(sat_image != 0).sel(band=['nir'])
            # histogram equalization
            nir.values = equalize(nir)
            
            # rasterize buildings
            building_raster = rasterize_buildings(sat_image, building_map)
            
            # create dictionary with sat_image data for plotting
            image_plot = {'Buildings overlay':building_map, 'Buildings':building_map,'RGB':rgb, 'NIRGB':nirgb, 'R':r, 'G':g, 'B':b, 'NIR':nir}
            
            # dictionary with sat_image data for saving
            image_data = {'RGB':rgb.values, 'NIRGB':nirgb.values, 'R':r.values, 'G':g.values, 'B':b.values, 'NIR':nir.values, 'Buildings':building_raster}

            return image_plot, image_data
            
def run_acquisition():
    
    # run data acquisition
    for city in CITIES:
        get_openstreetmap(city)
        coord_bounds, building_map = get_buildings(city)
        get_sat_img(coord_bounds, city)
        plot, data = img_process(city, building_map)
        # save dictionary as pkl
        save_data(data, city)
        # call plot function:
        plot_image_data(plot)
    
    # create test data
    get_openstreetmap(TEST_CITY) # extract data for Berlin
    coord_bounds, building_map = get_buildings(TEST_CITY, coord_bounds=TEST_COORDS)
    get_sat_img(coord_bounds, TEST_CITY)
    plot, data = img_process(TEST_CITY, building_map)
    save_data(data, TEST_CITY)
    plot_image_data(plot)

