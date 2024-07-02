import openeo
from openeo.processes import ProcessBuilder
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from matplotlib.colors import ListedColormap, Normalize
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
def get_openstreetmap(city, osm_path=OSM_PATH):
    # create path if not there
    if not os.path.isdir(osm_path):
        os.makedirs(osm_path)
    # download OSM map
    osm_map_path = pyrosm.get_data(city, directory=osm_path)
    # return file path
    return osm_map_path

# read building geometry out of OSM
def get_buildings(city, osm_path=OSM_PATH, coord_bounds=None):
    if coord_bounds is None:
        osm_map = pyrosm.OSM(osm_path)
    else:
        osm_map = pyrosm.OSM(osm_path, coord_bounds)

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
        out_shape=(map.height, map.width),
        transform=map.transform, # TODO is that right?
        fill=0,
        all_touched=True, 
        dtype=rasterio.uint8
    )
    return building_raster

def plot_image_data(image_data):
    fig = plt.figure(figsize=(16,10))
    ax_list = fig.subplots(2,4)

    # create new customized colormap:[gradient from (red, green, blue, alpha), to (red, green, blue, alpha)]
    building_cmap = ListedColormap([(0, 0, 0, 0), (0.01, 0.18, 0.9, 1)])

    # plot overlapping buildings
    rgb = np.dstack([image_data['RGB'][0],image_data['RGB'][1],image_data['RGB'][2]])
    ax_list[0][0].imshow(rgb, zorder=1)
    ax_list[0][0].imshow(image_data['Buildings'], cmap=building_cmap, zorder=2)
    
    # plot buildings
    image_data['Buildings overlay'].plot(ax=ax_list[0][1])

    # plot RGB
    ax_list[0][2].imshow(rgb)

    # plot NIRGB
    nirgb = np.dstack([image_data['NIRGB'][0],image_data['NIRGB'][1],image_data['NIRGB'][2]])
    ax_list[0][3].imshow(nirgb)

    # plot singe channels
    ax_list[1][0].imshow(image_data['R'], cmap='gray')
    ax_list[1][1].imshow(image_data['G'], cmap='gray')
    ax_list[1][2].imshow(image_data['B'], cmap='gray')
    ax_list[1][3].imshow(image_data['NIR'], cmap='gray')

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
    print(f'Image data {city} written.')

def color_correct(channels,r_cor =1, g_cor=1, b_cor=1):
    return channels[0]*r_cor, channels[1]*g_cor, channels[2]*b_cor

def equalize(channel):
    values = channel.copy()
    # replace nan values by 0
    values = np.nan_to_num(values)
    return exposure.equalize_hist(values)

def reproject_crs(file_path, building_map):
    '''
    Project the satelite image on the building map
    '''
    # calculate transformation, width and height of the reprojction
    with rasterio.open(file_path) as sat_image:
        transform, width, height = calculate_default_transform(sat_image.crs, building_map.crs, sat_image.width, sat_image.height,*sat_image.bounds)
        # update metadata 
        kwargs = sat_image.meta.copy()
        kwargs.update({
            'crs':building_map.crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # create destination image as rasterio dataset
        sat_image_repro = MemoryFile()

        # project sat_image on the building_map
        with sat_image_repro.open(**kwargs) as dst:
            for i in range(1, sat_image.count + 1):
                reproject(
                    source=rasterio.band(sat_image, i),
                    # insert directly into numpy ndarray
                    destination=rasterio.band(dst, i),
                    src_transform=sat_image.transform,
                    src_crs=sat_image.crs,
                    dst_transform=transform,
                    dst_crs=building_map.crs,
                    resampling=Resampling.nearest)

    return sat_image_repro

def img_process(city, building_map, path=IMAGE_DATA_PATH):
    
    file_path = os.path.join(path, f'{city}.tif')

    # match sat_image CRS
    #building_map = building_map.to_crs(sat_image.crs)
    sat_image_repro = reproject_crs(file_path, building_map)

    with sat_image_repro.open() as sat_image:
        # create rgb
        rgb = sat_image.read([3, 2, 1])
        # histogram equalization: rgb.values gives the np.array
        rgb = equalize(rgb)
        # manually correct color values
        #rgb.values[0],rgb.values[1], rgb.values[2] = color_correct(rgb.values)
        
        # create nirgb
        nirgb = sat_image.read([4, 2, 1])
        # histogram equalization
        nirgb= equalize(nirgb)
        
        # r
        r = sat_image.read(3)
        # histogram equalization
        r = equalize(r)

        # g
        g = sat_image.read(2)
        # histogram equalization
        g = equalize(g)

        # b
        b = sat_image.read(1)
        # histogram equalization
        b = equalize(b)

        # nir
        nir = sat_image.read(4)
        # histogram equalization
        nir = equalize(nir)
        
        # rasterize buildings
        building_raster = rasterize_buildings(sat_image, building_map)
        
        # create dictionary with sat_image data for plotting
        image_plot = {'Buildings overlay':building_map, 'Buildings':building_raster,'RGB':rgb, 'NIRGB':nirgb, 'R':r, 'G':g, 'B':b, 'NIR':nir}
        
        # dictionary with sat_image data for saving
        image_data = {'RGB':rgb, 'NIRGB':nirgb, 'R':r, 'G':g, 'B':b, 'NIR':nir, 'Buildings':building_raster}

        return image_plot, image_data
                
def run_acquisition():
    
    # run data acquisition
    for city in CITIES:
        osm_path=get_openstreetmap(city)
        # pass osm filepath to read buildings
        coord_bounds, building_map = get_buildings(city, osm_path=osm_path)
        # only if not yet downloaded
        if not os.path.exists(os.path.join(IMAGE_DATA_PATH, f'{city}.tif')) : get_sat_img(coord_bounds, city)
        plot, data = img_process(city, building_map)
        # save dictionary as pkl
        save_data(data, city)
        # call plot function:
        #plot_image_data(plot)
    
    # create test data
    get_openstreetmap(TEST_CITY) # extract data for Berlin
    coord_bounds, building_map = get_buildings(TEST_CITY, coord_bounds=TEST_COORDS)
    get_sat_img(coord_bounds, TEST_CITY)
    plot, data = img_process(TEST_CITY, building_map)
    save_data(data, TEST_CITY)
    #plot_image_data(plot)

