"""1.1 Data Aqcuisition and Alignment:
Module to acquire satellite image data from European cities and reprojects
them into equal earth properties with building maps. Plots serveral bands
and prepares the dataset for building segmentation."""

import os
import pickle
import openeo
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio.io import MemoryFile
import rasterio
import pyrosm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np

import params


def get_openstreetmap(city, osm_path=params.OSM_PATH):
    """
    Function to acquire Open Street Map file for the city and save it in the provided path.
    If the path does not exist, it creates it.

    Args:
        city (str): name of the city to acquire OSM map for
        osm_path (str): path where to save to OSM map file

    Returns:
        str: path where to save to OSM map file
    """

    # create path if not there
    if not os.path.isdir(osm_path):
        os.makedirs(osm_path)
    # download OSM map
    pyrosm.get_data(city, directory=osm_path)


def get_buildings(city, osm_path=params.OSM_PATH, coord_bounds=None):
    """
    Function to aquire the building geomerty from the OSM map file.

    Args:
        city (str): name of the city to acquire building geometry for
        osm_path (str): path where to save to buidling geometry file
        coord_bounds (tuple): coordinate bounds of the building geometry
            to extract, format: (min logitude, min latitude, max longitude, max latitude)

    Returns:
        tuple:
            - (tuple): coordinate bounds of the building geometry
    """

    osm_city_path = os.path.join(osm_path, f"{city}.osm.pbf")

    # if no coordinate bounds are given
    if coord_bounds is None:
        osm_map = pyrosm.OSM(osm_city_path)

    # extract by coordinate bounds
    else:
        osm_map = pyrosm.OSM(osm_city_path, coord_bounds)
    building_map = osm_map.get_buildings()

    # print message
    print(f"Building map for {city} created.")
    coord_bounds = building_map.geometry.total_bounds

    return coord_bounds, building_map


def get_sat_img(coord_bounds, city, image_path=params.IMAGE_DATA_PATH):
    """
    Function to acquire Sentinel 2 satellite images in bands R, G, B, NIR
    for the provided city name and saving them into the image_path as GTiff.

    Args:
        coord_bounds (tuple): coordinate bounds to extract images from
        city (str): name of the city to acquire building geometry for
        image_path (str): path where to save the satellite image data (default: params.IMAGE_DATA_PATH)

    Returns:
        None
    """

    # create sat_image path if not there:
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # get connection
    connection = openeo.connect("https://openeocloud.vito.be/openeo/1.0.0")

    # authentication: connection.authenticate_basic("username", "password")
    connection.authenticate_oidc()

    sat_data = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={
            "west": coord_bounds[0],
            "south": coord_bounds[1],
            "east": coord_bounds[2],
            "north": coord_bounds[3],
        },
        temporal_extent=["2023-03-01", "2023-07-01"],
        max_cloud_cover=30,
        bands=["B02", "B03", "B04", "B08"],  # B, G, R, infrared
    )

    # get mean of bands
    sat_data_mean = sat_data.mean_time()

    # save sat_image data
    sat_result = sat_data_mean.save_result(format="GTiff")
    file_name = os.path.join(image_path, f"{city}.tif")
    job = sat_result.create_job()
    job.start_and_wait()
    job.get_results().download_file(file_name)


def rasterize_buildings(base_map, building_map):
    """
    Rasterizs building geometry.

    Args:
        map(raserio.DatasetReader): satellite image (reprojected) as base
        building_map (geopandas.GeoDataFrame): GeodataFrame containing the building geometry

    Returns:
        numpy.ndarray: raserized building geometry
    """

    building_raster = rasterize(
        shapes=building_map.geometry,
        out_shape=(base_map.height, base_map.width),
        transform=base_map.transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8,
    )
    return building_raster


def plot_image_data(image_data, city):
    """
    Plotting image data for a given city: overlapping buildings, buildings, rgb, nirgb, r,g,b,
    and saves the plot to disk.

    Args:
        image_data (dict):
            - 'R': red channel
            - 'G': green channel
            - 'B': blue channel
            - 'NIR': near-infrared channel
            - 'Buildings': buildings labels

    Returns:
        None
    """

    fig = plt.figure(figsize=(32, 20))
    ax_list = fig.subplots(2, 4)

    # create new customized colormap:[gradient from (red, green, blue, alpha), to (red, green, blue, alpha)]
    building_cmap = ListedColormap([(0, 0, 0, 0), (0.01, 0.18, 0.9, 1)])

    rgb = np.dstack([image_data["RGB"][0], image_data["RGB"][1], image_data["RGB"][2]])
    nirgb = np.dstack(
        [image_data["NIRGB"][0], image_data["NIRGB"][1], image_data["NIRGB"][2]]
    )

    # plot overlapping buildings
    ax_list[0][0].imshow(rgb, zorder=1, aspect="equal")
    ax_list[0][0].imshow(
        image_data["Buildings"], cmap=building_cmap, zorder=2, aspect="equal"
    )

    # plot buildings
    ax_list[0][1].imshow(image_data["Buildings"], cmap=building_cmap)

    # plot RGB
    ax_list[0][2].imshow(rgb)

    # plot NIRGB
    ax_list[0][3].imshow(nirgb)

    # plot singe channels
    ax_list[1][0].imshow(image_data["R"], cmap="gray")
    ax_list[1][1].imshow(image_data["G"], cmap="gray")
    ax_list[1][2].imshow(image_data["B"], cmap="gray")
    ax_list[1][3].imshow(image_data["NIR"], cmap="gray")

    titles = ["Buildings overlay", "Buildings", "RGB", "NIRGB", "R", "G", "B", "NIR"]

    # adjust plot
    for title, ax in zip(titles, ax_list.flatten()):
        ax.axis("off")
        ax.set_aspect("equal")
        ax.margins(x=0)
        ax.margins(y=0)

        # save subplot figure
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(
            os.path.join(params.OUT_PATH, f"{city}_{title}.png"), bbox_inches=extent
        )

        # set title
        ax.set_title(title)
        ax.title.set_fontsize(14)

    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    # show plot
    plt.show()


def save_data(image_data, city, path=params.IMAGE_DATA_PATH):
    """
    Saves the image_data dictionary as .pkl to file.

    Args:
        image_data (dict):
            - 'R': red channel
            - 'G': green channel
            - 'B': blue channel
            - 'NIR': near-infrared channel
            - 'Buildings': buildings labels

    Returns:
        None
    """
    # save image dictionary as pkl
    file_name = f"{city}.pkl"

    with open(os.path.join(path, file_name), "wb") as file:
        pickle.dump(image_data, file)
    print(f"Image data {city} written.")


def equalize(channel):
    """
    Applies channel-wise histogram equalization.

    Agrs:
        channel (np.ndarray): color channel image

    Returns:
        np.ndarray:
    """
    values = channel.copy()
    # replace nan values by 0
    values = np.nan_to_num(values)

    return exposure.equalize_hist(values)


def reproject_crs(file_path, building_map):
    """
    Project the satelite image on the building map. Opens the satellite image from file.

    Args:
        file_path (str): path to satellite image file
        building_map (GeoDataFrame): GeoDataFrame of building geometry

    Returns:
        MemoryFile: reprojected satellite images
    """
    # calculate transformation, width and height of the reprojction
    with rasterio.open(file_path) as sat_image:
        transform, width, height = calculate_default_transform(
            sat_image.crs,
            building_map.crs,
            sat_image.width,
            sat_image.height,
            *sat_image.bounds,
        )

        # update metadata
        kwargs = sat_image.meta.copy()
        kwargs.update(
            {
                "crs": building_map.crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

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
                    resampling=Resampling.nearest,
                )

    return sat_image_repro


def img_process(city, building_map, path=params.IMAGE_DATA_PATH):
    """
    Extracts and equalizes channels from GTiff satellite image data, applies reprojection
    and returns a dictionary with single (combined) channels.

    Args:
        city (str): name of the city to acquire OSM map for
        building_map (geopandas.GeoDataFrame): GeodataFrame containing the building geometry
        path (str): path of the satellite GTiff file

    Returns:
        dict:
            - 'RGB': rgb channel
            - 'NIRGB': nirgb channel
            - 'R': red channel
            - 'G': green channel
            - 'B': blue channel
            - 'NIR': near-infrared channel
            - 'Buildings': buildings labels
    """

    file_path = os.path.join(path, f"{city}.tif")

    # match sat_image to building_map CRS
    sat_image_repro = reproject_crs(file_path, building_map)

    with sat_image_repro.open() as sat_image:
        # create rgb
        rgb = sat_image.read([3, 2, 1])
        # histogram equalization: rgb.values gives the np.array
        rgb = equalize(rgb)

        # create nirgb
        nirgb = sat_image.read([4, 2, 1])
        # histogram equalization
        nirgb = equalize(nirgb)

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

        # dictionary with sat_image data for saving
        return {
            "RGB": rgb,
            "NIRGB": nirgb,
            "R": r,
            "G": g,
            "B": b,
            "NIR": nir,
            "Buildings": building_raster,
        }


def run_acquisition(plot=False):
    """
    Conducts the data acquisition by downloading satellite data and OSM building maps for all
    the cities in the parameter list. Applies reprojection to the satellite data according to
    the OSM map, opionally plots and saves the data packet of individual channels as .pkl.

    Args:
        plot (bool): plots the image data per city if set True

    Returns:
        None
    """

    # run data acquisition
    for city in params.DATA_CITIES:
        # download OSM map if not available
        if not os.path.exists(os.path.join(params.OSM_PATH, f"{city}.osm.pbf")):
            get_openstreetmap(city)

        coord_bounds, building_map = get_buildings(city)

        # only if not yet downloaded
        if not os.path.exists(os.path.join(params.IMAGE_DATA_PATH, f"{city}.tif")):
            get_sat_img(coord_bounds, city)

        data = img_process(city, building_map)
        # save dictionary as pkl
        save_data(data, city)

        if plot:
            # call plot function:
            plot_image_data(data, city)

    # create test data
    get_openstreetmap(params.TEST_CITY)  # extract data for Berlin
    coord_bounds, building_map = get_buildings(
        params.TEST_CITY, coord_bounds=params.TEST_COORDS
    )

    # obtain satellite image if not yet there
    if not os.path.exists(
        os.path.join(params.IMAGE_DATA_PATH, f"{params.TEST_CITY}.tif")
    ):
        get_sat_img(coord_bounds, params.TEST_CITY)

    data = img_process(params.TEST_CITY, building_map)
    save_data(data, params.TEST_CITY)

    if plot:
        # call plot function:
        plot_image_data(data, params.TEST_CITY)
