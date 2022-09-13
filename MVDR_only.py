import numpy as np
from mm_process import process
import math
import os
from doatools.doatools import model, estimation
import matplotlib.pyplot as plt
import doatools.doatools.plotting as doaplt
from doatools.doatools.model.sources import FarField2DSourcePlacement

def clutter_removal(input_val, axis=0):  #
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean
    return output_val.transpose(reordering)

raw_mmWave = np.load('./realtime/Raw_data/test/mmWave.npy')
# raw_mmWave = np.load('./realtime/Alexander/data.npy')
# raw_mmWave = np.transpose(raw_mmWave, (1, 2, 0, 3))
# raw_mmWave = raw_mmWave[np.newaxis, ...]

# parameters
num_chirp = 128
num_sample = 256
frames_count = 10
# frames_count = len(raw_mmWave)
start_frame = 40
step_frame = 40
end_frame = start_frame + (frames_count - 1) * step_frame + 1

# file
store_path = './realtime/'
store_file = 'MyData'

pc_coordinate = []

for frame in range(start_frame, end_frame, step_frame):
# for frame in range(frames_count):
    print(frame)
    data = raw_mmWave[frame]
    data = data.reshape(-1, num_chirp, num_sample)
    rangeFFTResult = np.fft.fft(data)
    rangeFFTResult = clutter_removal(rangeFFTResult, axis=1)

    # define antenna
    wavelength = 3e8 / 60e9
    spacing = wavelength / 2.0
    # vir_ant = np.array([
    #     [3, 2, 2, 3, 1, 0, 0, 1, 3, 2, 2, 3],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
    # ]).T * spacing
    # vir_ant[:, 0] = -vir_ant[:, 0]  # lefthand to righthand coor
    vir_ant = np.array([
        [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 2, 2, 3, 1, 0, 0, 1, 3, 2, 2, 3]
    ]).T * spacing
    vir_ant[:, 2] = -vir_ant[:, 2]  # lefthand to righthand coor
    AntennaArr = model.ArrayDesign(vir_ant, name='IWR6843ISK-ODS')

    # MVDR
    # az_start, az_end = 30, 150
    # el_start, el_end = -60, 60
    az_start, az_end = 20, 141  # end not included
    el_start, el_end = -80, 71
    az_step = 1
    el_step = 1
    az_size = int((az_end - az_start) / az_step)
    el_size = int((el_end - el_start) / el_step)
    grid = estimation.FarField2DSearchGrid(start=(az_start, el_start),stop=(az_end, el_end), unit='deg', size=(az_size, el_size))
    estimator = estimation.MVDRBeamformer(AntennaArr, wavelength, grid)

    point_cloud = np.zeros((az_size, el_size, 75))
    for s in range(75):
        temp = rangeFFTResult[..., s] @ (rangeFFTResult[..., s].conj().T) / rangeFFTResult[..., s].shape[-1]
        resv, est, point_cloud[..., s] = estimator.estimate(temp, 1, return_spectrum=True)
        # resv, est, point_cloud[..., s] = estimator.estimate(np.cov(rangeFFTResult[..., s]), 1, return_spectrum=True)
    # for s in range(16):
    #     ax = plt.subplot(4, 4, s + 1)
    #     doaplt.plot_spectrum(point_cloud[..., 255 - s], grid, ax=ax)
    #     ax.set_title(str(255 - s))
    # plt.show()

    idx = np.argsort(point_cloud, axis=None)[::-1]
    az, el, d = np.unravel_index(idx, point_cloud.shape)
    az = az * az_step + az_start
    el = el * el_step + el_start
    # x = (d + 1) * np.cos(el * np.pi / 180.0) * np.cos(az * np.pi / 180.0)
    # y = (d + 1) * np.cos(el * np.pi / 180.0) * np.sin(az * np.pi / 180.0)
    # z = (d + 1) * np.sin(el * np.pi / 180.0)
    z = d * np.sin(el * np.pi / 180.0)
    x = d * np.cos(el * np.pi / 180.0) * np.cos(az * np.pi / 180.0)
    y = d * np.cos(el * np.pi / 180.0) * np.sin(az * np.pi / 180.0)

    power = np.log(point_cloud[((az - az_start) / az_step).astype(int), ((el - el_start) / el_step).astype(int), d])
    power_idx = np.argsort(power)[::-1]
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    points = np.concatenate((x,y,z), axis=1)

    power = power[power_idx]
    points = points[power_idx]
    y = points[:, 1]
    z = points[:, 2]
    x = points[:, 0]
    matching_idx = np.where((np.abs(x) < 20) & (y > 10) & (y < 20) & (z > -10) & (z < 25))
    # matching_idx = np.where((np.abs(x) < 50) & (y > 5) & (y < 25) & (z > -20) & (z < 50))
    points = points[matching_idx]
    print(points.shape)

    center_count = 10
    surround_center_count = 4000
    cluster_count = 0
    center = points[0]
    cluster = []
    while cluster_count != center_count:
        distances = np.sqrt(np.sum(np.power((points - center), 2), axis=1))
        distances_idx = np.argsort(distances, axis=None)[: surround_center_count]

        temp_cluster = points[distances_idx]
        temp_cluster_power = power[distances_idx].reshape((-1, 1))
        temp_cluster = np.concatenate((temp_cluster, temp_cluster_power), axis=1)

        indices = np.setxor1d(np.arange(points.shape[0]),distances_idx)
        points = points[indices]
        power = power[indices]

        power_idx = np.argsort(power)[::-1]
        points = points[power_idx]
        power = power[power_idx]

        center = points[0]

        cluster.append(temp_cluster)
        cluster_count += 1
        print("cluster:%d"%cluster_count)
    cluster = np.concatenate(cluster, axis=0)

    # power = power[matching_idx].reshape((-1, 1))
    # points = points[:10000]
    # power = power[:10000]
    # print(points.shape)
    # print(power.shape)
    # cluster = np.concatenate((points, power), axis=1)

    pc_coordinate.append(cluster)

print("Saving it to " + store_path + store_file + ".dbf/")
if not os.path.exists(store_path + store_file + ".dbf"):
    os.mkdir(store_path + store_file + ".dbf")
np.save(store_path + store_file + ".dbf/" + store_file + ".dbf.coordinate", np.array(pc_coordinate))
