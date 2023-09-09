from builtins import max
from threading import local

from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#%% information of the solar plant
global_longitude = 98.5
global_latitude = 39.4
global_altitude = 3000 # meters

#%% multithreads dealing system
from concurrent.futures import ProcessPoolExecutor

def split_dataframe(df, num_splits, workers, process_func):
    """
    Split a DataFrame into arbitrary parts and process them using multiple processes.

    Args:
        df (pd.DataFrame): The DataFrame to split and process.
        num_splits (int): The number of splits to create.
        process_func (callable): A function that processes a DataFrame segment.

    Returns:
        list: A list of results returned by the processing function for each segment.
    """
    # Calculate the number of rows in each split
    split_size = len(df) // num_splits

    # Create a list to store futures and results
    futures = []

    # Use ProcessPoolExecutor to parallelize processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(num_splits):
            # Calculate the start and end indices for each split
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else len(df)

            # Extract the DataFrame segment
            df_segment = df.iloc[start_idx:end_idx]

            # Submit the segment for processing
            future = executor.submit(process_func, df_segment)
            futures.append(future)

    # Wait for all futures to complete and collect results
    results = [future.result() for future in futures]

    return results

#%% some function about the sun
@lru_cache(maxsize=2000)
def solar_zenith_angle(latitude_deg, declination, hour_angle):
    """
    Calculate the solar zenith angle.(太阳高度角)

    :param latitude: Latitude of the location in degrees.
    :param declination: Solar declination angle in radian.
    :param hour_angle: Hour angle in radian.
    :return: Solar zenith angle in radian.
    """
    # Convert latitude from degrees to radians
    latitude = math.radians(latitude_deg)

    # Calculate SZA using the formula
    sza = (
        math.asin(
            math.sin(latitude)
          * math.sin(declination)
          + math.cos(latitude)
          * math.cos(declination)
          * math.cos(hour_angle)
        )
    )

    if sza < 0: return 0
    return sza

def value_cos2b_to_value_cosb(cos_2b):
    return np.sqrt((cos_2b + 1) / 2)

from datetime import datetime

def hour_minute_to_decimal_time(hour, minute):
    """
    Convert hour and minute to decimal time.

    :param hour: Hour of the day (0-23).
    :param minute: Minute of the hour (0-59).
    :return: Decimal time (e.g., 12:30 is 12.5).
    """
    decimal_time = hour + minute / 60.0
    return decimal_time

@lru_cache(maxsize=2000)
def solar_declination(day_of_year):
    """
    Calculate the solar declination angle.(赤纬角)

    :param day_of_year: Day of the year (1 for January 1st, 365 for December 31st).
    :return: Solar declination angle in radian.
    """
    n = day_of_year
    delta = math.asin(math.sin(math.radians(23.45))
          * math.sin(math.radians(360 * (284 + n) / 365)))
    return delta

@lru_cache(maxsize=2000)
def hour_angle(local_time):
    """
    Calculate the hour angle.(时角)

    :param local_time: the local time (e.g., 12.5 for 12:30).
    :return: Hour angle in radian.
    """
    H = (np.pi / 12.0) * (local_time - 12)
    return H


@lru_cache(maxsize=2000)
def day_of_year_from_date(year, month, day):
    """
    Calculate the day of the year (DOY) from a given year, month, and day.

    :param year: Year (e.g., 2023).
    :param month: Month (1-12).
    :param day: Day (1-31, depending on the month).
    :return: Day of the year (1 for January 1st, 365 for December 31st).
    """
    try:
        date_obj = datetime(year, month, day)
        day_of_year = date_obj.timetuple().tm_yday
        return day_of_year
    except ValueError:
        return -1  # Invalid date


@lru_cache(maxsize=2000)
def solar_azimuth_angle(latitude_deg, zenith_angle):
    """
    Calculate the azimuth angle.(方位角)

    :param latitude_deg: Latitude of the location in degrees.
    :param zenith_angle: Solar zenith angle in radians.
    :return: Azimuth angle in radians.
    """
    # Convert latitude from degrees to radians
    latitude = math.radians(latitude_deg)

    # Calculate azimuth angle using the formula
    azimuth_angle = math.acos(
        (np.sin(latitude) - np.sin(zenith_angle) * np.sin(latitude))
        /(np.cos(zenith_angle) * np.cos(latitude))
    )

    return azimuth_angle


#%% read data
df = pd.read_excel("auxa.xlsx")
df['z'] = 4

#%% Create a scatter plot
plt.scatter(df.x, df.y, label='Heliostat', color='blue', marker='o', s=4)

#%% Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heliostat positions')

#%% Add a legend and adjust it to 1:1
plt.legend()
plt.axis('equal')

#%% Display the plot
#plt.show()

#%% Save the plot
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')

plt.close()

#%% Distance to the origin point
df['distance'] = ((df.x**2 + df.y**2)**0.5)
df.distance = df.distance.astype(int)
distances = df.distance.unique()

#%% Count the number of heliostats at each distance
counts = []
for distance in distances:
    counts.append(len(df[df.distance == distance]))

# bind the counts and the distance into one dataframe
df_distances_counts = pd.DataFrame({'distance': distances, 'count_val': counts})
df_distances_counts['distancedistract3point5'] = df_distances_counts.distance - 3.5

# Calculate the pitch angle of each heliostat
df_distances_counts['reflect_with_ground_angle_r'] = np.arctan(80.0 / df_distances_counts.distancedistract3point5)
df_distances_counts['reflect_with_ground_angle_d'] = df_distances_counts.reflect_with_ground_angle_r * 180 / np.pi

#%% vector from helio to receiver
# (0, 0, 84) is the receiver position
df['vector_h2r_x'] = 0 - df.x
df['vector_h2r_y'] = 0 - df.y
df['vector_h2r_z'] = 84 - df.z
df['vector_h2r_len'] = (df.vector_h2r_x**2 + df.vector_h2r_y**2 + df.vector_h2r_z**2)**0.5

# unitilize the vector
df['vector_h2r_x_unit'] = df.vector_h2r_x / df.vector_h2r_len
df['vector_h2r_y_unit'] = df.vector_h2r_y / df.vector_h2r_len
df['vector_h2r_z_unit'] = df.vector_h2r_z / df.vector_h2r_len

#atmospheric attenuation
df['atmospheric_attenuation'] = 0.99321 - 0.0001176 * df.vector_h2r_len + 1.97e-8 * df.vector_h2r_len**2

#%% vector from helio to sun
def vector_h2s_x(day_of_year, local_time):
    decli = solar_declination(day_of_year)
    hourang = hour_angle(local_time)
    zenith = solar_zenith_angle(global_latitude, decli, hourang)
    azimuth = solar_azimuth_angle(global_latitude, zenith)
    return np.cos(zenith) * np.cos(azimuth)

def vector_h2s_y(day_of_year, local_time):
    decli = solar_declination(day_of_year)
    hourang = hour_angle(local_time)
    zenith = solar_zenith_angle(global_latitude, decli, hourang)
    azimuth = solar_azimuth_angle(global_latitude, zenith)
    return np.cos(zenith) * np.sin(azimuth)

def vector_h2s_z(day_of_year, local_time):
    decli = solar_declination(day_of_year)
    hourang = hour_angle(local_time)
    zenith = solar_zenith_angle(global_latitude, decli, hourang)
    return np.sin(zenith)

@lru_cache(maxsize=1000)
def vector_of_heliostat_to_sun_unit(day_of_year, local_time):
    h2s_x = vector_h2s_x(day_of_year, local_time)
    h2s_y = vector_h2s_y(day_of_year, local_time)
    h2s_z = vector_h2s_z(day_of_year, local_time)

    norm = (h2s_x**2 + h2s_y**2 + h2s_z**2)**0.5
    return np.array([h2s_x / norm, h2s_y / norm, h2s_z / norm]).T

@lru_cache(maxsize=2000)
def cosine_effiency_of_each_heliostat(day_of_year, local_time):
    vec_h2s_unit = vector_of_heliostat_to_sun_unit(day_of_year, local_time)
    result = pd.DataFrame()
    result['cosine_eff'] = value_cos2b_to_value_cosb(
        np.dot(
            df[['vector_h2r_x_unit', 'vector_h2r_y_unit', 'vector_h2r_z_unit']],
            vec_h2s_unit
        )
    )
    return result['cosine_eff']

@lru_cache(maxsize=2000)
def direct_normal_irradiance(day_of_year, local_time, altitude, latitude):
    """
    :param day_of_year:
    :param local_time:
    :param altitude: altitude of the target position (m)
    :return kW/m^2:
    """
    g0 = 1.366 # kW/m^2
    a = 0.4237 - 0.00821 * (6   - altitude/1000)**2
    b = 0.5055 + 0.00595 * (6.5 - altitude/1000)**2
    c = 0.2711 + 0.01858 * (2.5 - altitude/1000)**2

    declina = solar_declination(day_of_year)
    hourang = hour_angle(local_time)
    zenith = solar_zenith_angle(latitude, declina, hourang)
    return g0 * (a + b * np.exp(-c/np.sin(zenith))) if zenith != 0 else 0


#%% calculate the full reflective power
def full_reflective_power(day_of_year, local_time, heliostat_area):
    """
    :param day_of_year:
    :param local_time:
    :param heliostat_area:
    :return: the power of full reflective (kW)
    """
    cos_eff = cosine_effiency_of_each_heliostat(day_of_year, local_time)
    return (cos_eff
            * heliostat_area
            * direct_normal_irradiance(
                day_of_year,
                local_time,
                global_altitude,
                global_latitude))


#%% Calculate the yearly average cosine effiency
results = []
for month in range(1, 12 + 1):
    doy = day_of_year_from_date(2023, month, 21)
    local_time = []
    local_time.append(9)
    local_time.append(10.5)
    local_time.append(12)
    local_time.append(13.5)
    local_time.append(15)

    total_cos_eff = 0
    for t in local_time:
        total_cos_eff += cosine_effiency_of_each_heliostat(doy, t).mean()

    avg_cos_eff = total_cos_eff / len(local_time)
    results.append(avg_cos_eff)

#%% other tests (ready to delete)
example_doy = day_of_year_from_date(2023, 6, 21)
example_time = 12

#%% the function of the normal vector of the heliostat
def normalize_vector(row):
    magnitude = np.linalg.norm(row)  # Calculate the magnitude of the vector
    if magnitude != 0:
        return row / magnitude  # Normalize the vector
    else:
        return row  # Avoid division by zero for zero vectors

@lru_cache(maxsize=1000)
def normal_vector_of_heliostat(doy, local_time):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :return: the normal vectors of each of the heliostat
    """
    vec_h2s = vector_of_heliostat_to_sun_unit(doy, local_time)
    normal = df[['vector_h2r_x_unit', 'vector_h2r_y_unit', 'vector_h2r_z_unit']] + vec_h2s
    normal = normal.apply(normalize_vector, axis=1)
    new_column_names = {'vector_h2r_x_unit': 'normal_x_unit',
                        'vector_h2r_y_unit': 'normal_y_unit',
                        'vector_h2r_z_unit': 'normal_z_unit'}
    normal.rename(columns=new_column_names, inplace=True)
    return normal

@lru_cache(maxsize=1000)
#%% Calculate the heliostat azimuth angle
def heliostat_azimuth_angle(doy, local_time):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :return: the azimuth angle of each of the heliostat
    """
    normal = normal_vector_of_heliostat(doy, local_time)
    azimuth = np.arctan(normal.normal_y_unit / normal.normal_x_unit)
    return azimuth

#%% Calculate the heliostat tilt angle
@lru_cache(maxsize=1000)
def heliostat_tilt_angle(doy, local_time):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :return: the tilt angle of each of the heliostat
    """
    normal = normal_vector_of_heliostat(doy, local_time)
    tilt = np.arctan(np.sqrt(normal.normal_x_unit**2 + normal.normal_y_unit**2) / normal.normal_z_unit)
    return tilt

@lru_cache(maxsize=1000)
def heliostat_steering_angle(doy, local_time):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :return: the steering angle of each of the heliostat
    """
    normal = normal_vector_of_heliostat(doy, local_time)
    return np.arctan(normal.normal_y_unit / normal.normal_x_unit)

#%% function calculate the 4 corners of the heliostats
def up_left_corner_local_coordinate(central_x, central_y, length, width):
    return central_x - 0.5 * length, central_y + 0.5 * width

def up_right_corner_local_coordinate(central_x, central_y, length, width):
    return central_x + 0.5 * length, central_y + 0.5 * width

def down_left_corner_local_coordinate(central_x, central_y, length, width):
    return central_x - 0.5 * length, central_y - 0.5 * width

def down_right_corner_local_coordinate(central_x, central_y, length, width):
    return central_x + 0.5 * length, central_y - 0.5 * width

@lru_cache(maxsize=2000)
def each_global_coordinate(local_x, local_y, doy, local_time, width, length):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :param width: the width of the heliostat
    :param length: the length of the heliostat
    :return: the global coordinate of the up left corner of the heliostat
    """
    tilt = heliostat_tilt_angle(doy, local_time)
    steering = heliostat_steering_angle(doy, local_time)

    central_x = df.x
    central_y = df.y
    central_z = df.z

    global_x = - np.sin(steering) * local_x - np.cos(steering) * np.cos(tilt) * local_y + central_x
    global_y = np.cos(steering) * local_x - np.sin(steering) * np.cos(tilt) * local_y + central_y
    global_z = np.sin(tilt) * local_y + central_z

    return global_x, global_y, global_z

def up_left_corner_global_coordinate(doy, local_time, width, length):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :param width: the width of the heliostat
    :param length: the length of the heliostat
    :return: the global coordinate of the up left corner of the heliostat
    """
    local_x, local_y = up_left_corner_local_coordinate(0, 0, length, width)
    return each_global_coordinate(local_x, local_y, doy, local_time, width, length)

def up_right_corner_global_coordinate(doy, local_time, width, length):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :param width: the width of the heliostat
    :param length: the length of the heliostat
    :return: the global coordinate of the up right corner of the heliostat
    """
    local_x, local_y = up_right_corner_local_coordinate(0, 0, length, width)
    return each_global_coordinate(local_x, local_y, doy, local_time, width, length)


def down_left_corner_global_coordinate(doy, local_time, width, length):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :param width: the width of the heliostat
    :param length: the length of the heliostat
    :return: the global coordinate of the down left corner of the heliostat
    """
    local_x, local_y = down_left_corner_local_coordinate(0, 0, length, width)
    return each_global_coordinate(local_x, local_y, doy, local_time, width, length)

def down_right_corner_global_coordinate(doy, local_time, width, length):
    """
    :param doy: day of year
    :param local_time: the current time of your time zone. (eg: 18:00 is 18)
    :param width: the width of the heliostat
    :param length: the length of the heliostat
    :return: the global coordinate of the down right corner of the heliostat
    """
    local_x, local_y = down_right_corner_local_coordinate(0, 0, length, width)
    return each_global_coordinate(local_x, local_y, doy, local_time, width, length)

#%% function of the cooridinate of the projection of each heliostat on the ground
@lru_cache(maxsize=2000)
def heliostat_corner_projection_on_ground(doy, local_time, width, length, corner_global_coordinate_func):
    zenith = solar_zenith_angle(global_latitude, solar_declination(doy), hour_angle(local_time))

    corner_x, corner_y, corner_z = corner_global_coordinate_func(doy, local_time, width, length)
    solar_azimuth = solar_azimuth_angle(global_latitude, zenith)

    x_on_the_ground = corner_x - corner_z * (1/np.tan(zenith)) * np.cos(solar_azimuth)
    y_on_the_ground = corner_y - corner_z * (1/np.tan(zenith)) * np.sin(solar_azimuth)

    return x_on_the_ground, y_on_the_ground

def projection_of_heliostat_on_the_ground(doy, local_time, width, length):
    c1_x, c1_y = heliostat_corner_projection_on_ground(doy, local_time, width, length, up_left_corner_global_coordinate)
    c2_x, c2_y = heliostat_corner_projection_on_ground(doy, local_time, width, length, up_right_corner_global_coordinate)
    c3_x, c3_y = heliostat_corner_projection_on_ground(doy, local_time, width, length, down_right_corner_global_coordinate)
    c4_x, c4_y = heliostat_corner_projection_on_ground(doy, local_time, width, length, down_left_corner_global_coordinate)

    c1 = list(zip(c1_x, c1_y))
    c2 = list(zip(c2_x, c2_y))
    c3 = list(zip(c3_x, c3_y))
    c4 = list(zip(c4_x, c4_y))

    result = pd.DataFrame()
    result['c1'] = c1
    result['c2'] = c2
    result['c3'] = c3
    result['c4'] = c4

    return result

#%% represent those projection polygon on the ground by shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

def polygon_of_heliostat_on_the_ground(doy, local_time, width, length):
    projection = projection_of_heliostat_on_the_ground(doy, local_time, width, length)
    return projection.apply(lambda x: Polygon(x), axis=1)

#%% overlapping parts of projections of heliostats on the ground
def overlappings_of_projection(polygons):
    overlappings = []
    for p in polygons:
        for q in polygons:
            if p != q:
                overlappings.append(p.intersection(q))
    unions = unary_union(overlappings)
    return unions

#%% shadowing effeciency
@lru_cache(maxsize=2000)
def shadowing_efficiency_of_the_whole_field(doy, local_time, width, length):
    zenith = solar_zenith_angle(global_latitude, solar_declination(doy), hour_angle(local_time))
    if zenith <= 0: return 0

    ps = polygon_of_heliostat_on_the_ground(doy, local_time, width, length)

    whole_proj = split_dataframe(
        ps, 60, 12, unary_union
    )
    whole_proj = unary_union(whole_proj)

    overlappings = split_dataframe(
        ps, 60, 12, overlappings_of_projection
    )
    overlappings = unary_union(overlappings)

    return 1 - overlappings.area / whole_proj.area

#%% tests
#ps = polygon_of_heliostat_on_the_ground(80, 17, 6, 6)
#overlappings = split_dataframe(
#    ps, 60, 12, overlappings_of_projection
#)
#overlappings = unary_union(overlappings)
#print(overlappings.area)
#print(unary_union(ps).area)
#print(shadowing_efficiency(80, 12, 6, 6))

#%% solar escaped height
def height_solar_escaped_ratio_of_each_heliostat(doy, local_time, width, length):
    # height of the lower side of heliostat
    local_x, local_y = down_left_corner_local_coordinate(0, 0, length, width)
    x, y, height = each_global_coordinate(local_x, local_y, doy, local_time, width, length)
    distance_to_tower = ((x - 3.5)**2 + (y - 3.5)**2)**0.5
    start = np.array([distance_to_tower, height])
    example_line = lambda x: - (80 / ((df.x**2 + df.y**2)**0.5)) * (x - distance_to_tower) + height
    end_height = example_line(0)
    solar_height = 84 - end_height
    escaped_height = 80 - end_height
    ratio = escaped_height / solar_height
    ratio[ratio < 0] = 0
    ratio[ratio > 1] = 1
    return ratio

#print(height_solar_escaped_ratio(80, 12, 6, 6))

#%% the turncate effiency
@lru_cache(maxsize=2000)
def turncate_efficiency_of_each_heliostat(doy, local_time, width, length):
    return (
        (1 - height_solar_escaped_ratio_of_each_heliostat(doy, local_time, width, length))
        / shadowing_efficiency_of_the_whole_field(doy, local_time, width, length)
    )

#%% the optical effiency of heliostats
@lru_cache(maxsize=1000)
def optical_efficiency_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity=0.92):
    return (turncate_efficiency_of_each_heliostat(doy, local_time, width, length)
            * shadowing_efficiency_of_the_whole_field(doy, local_time, width, length)
            * cosine_effiency_of_each_heliostat(doy, local_time)
            * df.atmospheric_attenuation
            * mirror_reflectivity)

#%% the power of each heliostat
@lru_cache(maxsize=1000)
def power_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity=0.92):
    return (optical_efficiency_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity)
            * direct_normal_irradiance(doy, local_time, global_altitude, global_latitude)
            * width * length)

#%% the power of the whole field
def power_of_the_whole_field(doy, local_time, width, length, mirror_reflectivity=0.92):
    return power_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity).sum()

#%% the power of the whole field per unit area
def power_of_the_whole_field_per_unit_area(doy, local_time, width, length, field_area, mirror_reflectivity=0.92):
    return power_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity).sum() / field_area

#%% some useful functions of heliostats
def total_area_of_heliostats(width, length):
    return width * length * len(df)

def number_of_heliostats():
    return len(df)

#%% question 1 solution

# dimension of the heliostats
width = 6
length = 6

# local_time
local_time = []
local_time.append(9)
local_time.append(10.5)
local_time.append(12)
local_time.append(13.5)
local_time.append(15)

# doy
doy = []
for month in range(1, 13):
    doy.append(day_of_year_from_date(2023, month, 21))

#with ProcessPoolExecutor(12) as executor:
# day average of each properities of the field
def cosine_eff_func():
    # cosine eff
    cos_eff = []
    for d in doy:
        cos_eff_per_day = []
        for t in local_time:
            cos_eff_per_day.append(cosine_effiency_of_each_heliostat(d, t).mean())
        cos_eff.append(np.average(cos_eff_per_day, axis=0))
        print("cos eff ", d, "day OK")
    return cos_eff

def shadowing_eff_func():
    # shadowing eff
    shadow_eff = []
    for d in doy:
        shadow_eff_per_day = []
        for t in local_time:
            shadow_eff_per_day.append(shadowing_efficiency_of_the_whole_field(d, t, width, length))
        shadow_eff.append(np.average(shadow_eff_per_day, axis=0))
        print("shadow eff ", d, "day OK")
    return shadow_eff

def turncate_eff_func():
    # turncate eff
    turncate_eff = []
    for d in doy:
        turncate_eff_per_day = []
        for t in local_time:
            per_day_val = turncate_efficiency_of_each_heliostat(d, t, width, length).mean()
            turncate_eff_per_day.append(per_day_val)
        turncate_eff.append(np.average(turncate_eff_per_day, axis=0))
        print("turncate eff ", d, "ok")
    return turncate_eff

def optical_eff_func():
    # optical eff
    optical_eff = []
    for d in doy:
        optical_eff_per_day = []
        for t in local_time:
            optical_eff_per_day.append(optical_efficiency_of_each_heliostat(d, t, width, length).mean())
        optical_eff.append(np.average(optical_eff_per_day, axis=0))
        print("opti eff ", d, "day OK")
    return optical_eff

def power_per_unit_func():
    # power of each heliostat per unit
    power_per_unit = []
    for d in doy:
        power_per_unit_per_day = []
        for t in local_time:
            power_per_unit_per_day.append(power_of_each_heliostat(d, t, width, length).mean())
        power_per_unit.append(np.average(power_per_unit_per_day, axis=0))
        print("power per unit ", d, "day OK")
    return power_per_unit

with ProcessPoolExecutor(max_workers=12) as executor:
    cos_eff = executor.submit(cosine_eff_func)
    shadow_eff = executor.submit(shadowing_eff_func)
    turncate_eff = executor.submit(turncate_eff_func)
    optical_eff = executor.submit(optical_eff_func)
    power_per_unit = executor.submit(power_per_unit_func)

# arrange those stuff into a DataFrame
df_q1 = pd.DataFrame()
df_q1['optical_eff'] = optical_eff.result()
print("opt ok")
df_q1['cos_eff'] = cos_eff.result()
print("cos ok")
df_q1['shadow_eff'] = shadow_eff.result()
print("sha ok")
df_q1['turncate_eff'] = turncate_eff.result()
print("turn ok")
df_q1['power_per_unit'] = power_per_unit.result()
print("power ok")

#%% draw a hot map while color scheme is blue by sns
# and I want the figure a little bigger which could give me a good print quality
import seaborn as sns
df_q1_2 = df_q1.drop('power_per_unit', axis=1)
sns.heatmap(df_q1_2, cmap='Blues', annot=True, fmt='.3f', linewidths=.5, annot_kws={"size": 10})
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('heliostat_properties.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

#%% save the df_q1 into a csv file
df_q1['month'] = range(1, 13)
df_q1.to_csv('heliostat_properties.csv')

#%% year average of each properities of the field
df_q1_year = df_q1.mean(axis=0)