from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

global_longitude = 98.5
global_latitude = 39.4
global_altitude = 3000 # meters

from concurrent.futures import ProcessPoolExecutor

def split_dataframe(df, num_splits, workers, process_func):
    split_size = len(df) // num_splits

    futures = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else len(df)
            df_segment = df.iloc[start_idx:end_idx]
            future = executor.submit(process_func, df_segment)
            futures.append(future)
    results = [future.result() for future in futures]

    return results

@lru_cache(maxsize=2000)
def solar_zenith_angle(latitude_deg, declination, hour_angle):
    latitude = np.radians(latitude_deg)
    sza = (
        np.arcsin(
            np.sin(latitude)
          * np.sin(declination)
          + np.cos(latitude)
          * np.cos(declination)
          * np.cos(hour_angle)
        )
    )

    if sza < 0: sza = 0
    return sza

def value_cos2b_to_value_cosb(cos_2b):
    return np.sqrt((cos_2b + 1) / 2)

from datetime import datetime

def hour_minute_to_decimal_time(hour, minute):
    decimal_time = hour + minute / 60.0
    return decimal_time

@lru_cache(maxsize=2000)
def solar_declination(day_of_year):
    n = day_of_year
    delta = math.asin(math.sin(math.radians(23.45))
          * math.sin(math.radians(360 * (284 + n) / 365)))
    return delta

@lru_cache(maxsize=2000)
def hour_angle(local_time):
    H = (np.pi / 12.0) * (local_time - 12)
    return H


@lru_cache(maxsize=2000)
def day_of_year_from_date(year, month, day):
    try:
        date_obj = datetime(year, month, day)
        day_of_year = date_obj.timetuple().tm_yday
        return day_of_year
    except ValueError:
        return -1  # Invalid date


@lru_cache(maxsize=2000)
def solar_azimuth_angle(latitude_deg, zenith_angle):
    latitude = math.radians(latitude_deg)

    azimuth_angle = math.acos(
        (np.sin(latitude) - np.sin(zenith_angle) * np.sin(latitude))
        /(np.cos(zenith_angle) * np.cos(latitude))
    )

    return azimuth_angle


df = pd.read_excel("auxa.xlsx")
df['z'] = 4

plt.scatter(df.x, df.y, label='Heliostat', color='blue', marker='o', s=4)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heliostat positions')

plt.legend()
plt.axis('equal')

plt.close()

df['distance'] = ((df.x**2 + df.y**2)**0.5)
df.distance = df.distance.astype(int)
distances = df.distance.unique()

counts = []
for distance in distances:
    counts.append(len(df[df.distance == distance]))

df_distances_counts = pd.DataFrame({'distance': distances, 'count_val': counts})
df_distances_counts['distancedistract3point5'] = df_distances_counts.distance - 3.5

df_distances_counts['reflect_with_ground_angle_r'] = np.arctan(80.0 / df_distances_counts.distancedistract3point5)
df_distances_counts['reflect_with_ground_angle_d'] = df_distances_counts.reflect_with_ground_angle_r * 180 / np.pi

df['vector_h2r_x'] = 0 - df.x
df['vector_h2r_y'] = 0 - df.y
df['vector_h2r_z'] = 84 - df.z
df['vector_h2r_len'] = (df.vector_h2r_x**2 + df.vector_h2r_y**2 + df.vector_h2r_z**2)**0.5

df['vector_h2r_x_unit'] = df.vector_h2r_x / df.vector_h2r_len
df['vector_h2r_y_unit'] = df.vector_h2r_y / df.vector_h2r_len
df['vector_h2r_z_unit'] = df.vector_h2r_z / df.vector_h2r_len

df['atmospheric_attenuation'] = 0.99321 - 0.0001176 * df.vector_h2r_len + 1.97e-8 * df.vector_h2r_len**2

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
    g0 = 1.366 # kW/m^2
    a = 0.4237 - 0.00821 * (6   - altitude/1000)**2
    b = 0.5055 + 0.00595 * (6.5 - altitude/1000)**2
    c = 0.2711 + 0.01858 * (2.5 - altitude/1000)**2

    declina = solar_declination(day_of_year)
    hourang = hour_angle(local_time)
    zenith = solar_zenith_angle(latitude, declina, hourang)
    return g0 * (a + b * np.exp(-c/np.sin(zenith))) if zenith != 0 else 0


def full_reflective_power(day_of_year, local_time, heliostat_area):
    cos_eff = cosine_effiency_of_each_heliostat(day_of_year, local_time)
    return (cos_eff
            * heliostat_area
            * direct_normal_irradiance(
                day_of_year,
                local_time,
                global_altitude,
                global_latitude))


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

example_doy = day_of_year_from_date(2023, 6, 21)
example_time = 12

def normalize_vector(row):
    magnitude = np.linalg.norm(row)  # Calculate the magnitude of the vector
    if magnitude != 0:
        return row / magnitude  # Normalize the vector
    else:
        return row  # Avoid division by zero for zero vectors

@lru_cache(maxsize=1000)
def normal_vector_of_heliostat(doy, local_time):
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
    normal = normal_vector_of_heliostat(doy, local_time)
    azimuth = np.arctan(normal.normal_y_unit / normal.normal_x_unit)
    return azimuth

@lru_cache(maxsize=1000)
def heliostat_tilt_angle(doy, local_time):
    normal = normal_vector_of_heliostat(doy, local_time)
    tilt = np.arctan(np.sqrt(normal.normal_x_unit**2 + normal.normal_y_unit**2) / normal.normal_z_unit)
    return tilt

@lru_cache(maxsize=1000)
def heliostat_steering_angle(doy, local_time):
    normal = normal_vector_of_heliostat(doy, local_time)
    return np.arctan(normal.normal_y_unit / normal.normal_x_unit)

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
    local_x, local_y = up_left_corner_local_coordinate(0, 0, length, width)
    return each_global_coordinate(local_x, local_y, doy, local_time, width, length)

def up_right_corner_global_coordinate(doy, local_time, width, length):
    local_x, local_y = up_right_corner_local_coordinate(0, 0, length, width)
    return each_global_coordinate(local_x, local_y, doy, local_time, width, length)


def down_left_corner_global_coordinate(doy, local_time, width, length):
    local_x, local_y = down_left_corner_local_coordinate(0, 0, length, width)
    return each_global_coordinate(local_x, local_y, doy, local_time, width, length)

def down_right_corner_global_coordinate(doy, local_time, width, length):
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

from shapely.geometry import Polygon
from shapely.ops import unary_union

def polygon_of_heliostat_on_the_ground(doy, local_time, width, length):
    projection = projection_of_heliostat_on_the_ground(doy, local_time, width, length)
    return projection.apply(lambda x: Polygon(x), axis=1)

def overlappings_of_projection(polygons):
    overlappings = []
    for p in polygons:
        for q in polygons:
            if p != q:
                overlappings.append(p.intersection(q))
    unions = unary_union(overlappings)
    return unions

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

@lru_cache(maxsize=2000)
def turncate_efficiency_of_each_heliostat(doy, local_time, width, length):
    return (
        (1 - height_solar_escaped_ratio_of_each_heliostat(doy, local_time, width, length))
        / shadowing_efficiency_of_the_whole_field(doy, local_time, width, length)
    )

@lru_cache(maxsize=1000)
def optical_efficiency_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity=0.92):
    return (turncate_efficiency_of_each_heliostat(doy, local_time, width, length)
            * shadowing_efficiency_of_the_whole_field(doy, local_time, width, length)
            * cosine_effiency_of_each_heliostat(doy, local_time)
            * df.atmospheric_attenuation
            * mirror_reflectivity)

@lru_cache(maxsize=1000)
def power_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity=0.92):
    return (optical_efficiency_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity)
            * direct_normal_irradiance(doy, local_time, global_altitude, global_latitude)
            * width * length)

def power_of_the_whole_field(doy, local_time, width, length, mirror_reflectivity=0.92):
    return power_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity).sum()

def power_of_the_whole_field_per_unit_area(doy, local_time, width, length, field_area, mirror_reflectivity=0.92):
    return power_of_each_heliostat(doy, local_time, width, length, mirror_reflectivity).sum() / field_area

def total_area_of_heliostats(width, length):
    return width * length * len(df)

def number_of_heliostats():
    return len(df)


width = 6
length = 6

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

#%% Calclation of the answer of question 1

def question1_solver():
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

    df_q1['month'] = range(1, 13)
    df_q1_year = df_q1.mean(axis=0)

    return df_q1, df_q1_year

#%% new global corridiante extension
def local_polar_to_global_rect_corridinate(polar_center_x, polar_center_y, polar_radius, polar_angle):
    return (
        polar_center_x + polar_radius * np.cos(polar_angle),
        polar_center_y + polar_radius * np.sin(polar_angle)
    )

def out_of_field(rect_x, rect_y):
    return (rect_x**2 + rect_y**2)**0.5 > 350

def heliostat_rect_corridinate_placing(tower_x, tower_y, receiver_z, distance_each_other, heliostat_height):
    polar_radius = []
    polar_angle = []
    max_heliostat_polar_radius = (tower_x**2 + tower_y**2)**0.5 + 350
    min_heliostat_polar_radius = 100
    heliostat_polar_radius = np.arange(min_heliostat_polar_radius, max_heliostat_polar_radius, distance_each_other)
    for r in heliostat_polar_radius:
        distance_angle = np.arcsin((distance_each_other) / r)
        heliostat_polar_angle = np.arange(0, 2*np.pi, distance_angle)
        if np.abs(heliostat_polar_angle[0] + 2*np.pi - heliostat_polar_angle[-1]) >= distance_angle:
            polar_radius.append(r)
            polar_angle.append(heliostat_polar_angle[0])
        for a in heliostat_polar_angle[1:]:
            polar_radius.append(r)
            polar_angle.append(a)

    df = pd.DataFrame({'r': polar_radius, 'a': polar_angle})
    df['x'], df['y'] = local_polar_to_global_rect_corridinate(tower_x, tower_y, df.r, df.a)
    df = df[~df.apply(lambda x: out_of_field(x.x, x.y), axis=1)]

    #other useful information of those heliostats
    df['z'] = heliostat_height
    df['distance_to_tower'] = df.r
    df['distance'] = df.r
    df['vector_h2r_x_unit'] = tower_x - df.x
    df['vector_h2r_y_unit'] = tower_y - df.y
    df['vector_h2r_z_unit'] = receiver_z - df.z
    df['len_h2r'] = (df.vector_h2r_x_unit**2 + df.vector_h2r_y_unit**2 + df.vector_h2r_z_unit**2)**0.5
    df['vector_h2r_x_unit'] = df.vector_h2r_x_unit / df.len_h2r
    df['vector_h2r_y_unit'] = df.vector_h2r_y_unit / df.len_h2r
    df['vector_h2r_z_unit'] = df.vector_h2r_z_unit / df.len_h2r
    df['atmospheric_attenuation'] = 0.99321 - 0.0001176 * df.len_h2r + 1.97e-8 * df.len_h2r** 2

    return df

def plot_heliostat_field(df):
    plt.scatter(df.x, df.y, label='Heliostat', color='blue', marker='o', s=4)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Heliostat positions')

    plt.legend()
    plt.axis('equal')
    plt.show()
    plt.close()

#%% test
df_new = heliostat_rect_corridinate_placing(-260, 40, 84, 12.777, 5)
plot_heliostat_field(df_new)
df = df_new

#%% question 2 solution
width = 5
length = 8
#new_sln_per_month, new_sln_per_year = question1_solver()

#%% critical seasonal time point
doy_spring_equinox = day_of_year_from_date(2023, 3, 21)
doy_summer_solstice = day_of_year_from_date(2023, 6, 21)
doy_autumn_equinox = day_of_year_from_date(2023, 9, 21)
doy_winter_solstice = day_of_year_from_date(2023, 12, 21)
time_duration = np.arange(4, 21, 0.1)

#%% plot the zenith angle function

#prepare the dataset
zenith_spring_equinox = []
zenith_summer_solstice = []
zenith_autumn_equinox = []
zenith_winter_solstice = []
for t in time_duration:
    zenith_spring_equinox.append(solar_zenith_angle(global_latitude, solar_declination(doy_spring_equinox), hour_angle(t)))
    zenith_summer_solstice.append(solar_zenith_angle(global_latitude, solar_declination(doy_summer_solstice), hour_angle(t)))
    zenith_autumn_equinox.append(solar_zenith_angle(global_latitude, solar_declination(doy_autumn_equinox), hour_angle(t)))
    zenith_winter_solstice.append(solar_zenith_angle(global_latitude, solar_declination(doy_winter_solstice), hour_angle(t)))

zenith_spring_df = pd.DataFrame()
zenith_spring_df['doy'] = [doy_spring_equinox for i in range(len(time_duration))]
zenith_spring_df['time'] = time_duration
zenith_spring_df['zenith'] = zenith_spring_equinox

zenith_summer_df = pd.DataFrame()
zenith_summer_df['doy'] = [doy_summer_solstice for i in range(len(time_duration))]
zenith_summer_df['time'] = time_duration
zenith_summer_df['zenith'] = zenith_summer_solstice

zenith_autumn_df = pd.DataFrame()
zenith_autumn_df['doy'] = [doy_autumn_equinox for i in range(len(time_duration))]
zenith_autumn_df['time'] = time_duration
zenith_autumn_df['zenith'] = zenith_autumn_equinox

zenith_winter_df = pd.DataFrame()
zenith_winter_df['doy'] = [doy_winter_solstice for i in range(len(time_duration))]
zenith_winter_df['time'] = time_duration
zenith_winter_df['zenith'] = zenith_winter_solstice

#%% function plot X, Y
def plot_zenith(zenith_df, name):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig, ax = plt.subplots()
    ax.plot(zenith_df.time, zenith_df.zenith, label='太阳高度角')
    ax.set_ylim(0, 1.7)
    plt.savefig(name)
    plt.close()

plot_zenith(zenith_spring_df, "spring")
plot_zenith(zenith_summer_df, "summer")
plot_zenith(zenith_autumn_df, "autumnn")
plot_zenith(zenith_winter_df, "winter")
