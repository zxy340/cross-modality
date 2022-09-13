import numpy as np
from mm_process import process
import math
import os
from doatools.doatools import model, estimation
import matplotlib.pyplot as plt
import doatools.doatools.plotting as doaplt
from doatools.doatools.model.sources import FarField2DSourcePlacement

# ................................................................
# num_chirp = 128
# num_sample = 256
# num_Tx = 3
# num_channel = 4
# dfile = './realtime/Raw_data/test/adc_data_Raw_0.bin'
# num_frame = os.path.getsize(dfile) // num_chirp // num_Tx // num_channel // num_sample // 2 // 2
# x = np.memmap(dfile, mode='r', dtype=np.int16, shape=(num_frame, num_chirp, num_Tx, num_channel, num_sample * 2))
# raw_data = x.reshape((-1, num_chirp*num_Tx*num_channel*num_sample*2))
# print(np.shape(raw_data))
# readItem = raw_data[50]
# readItem = readItem.reshape(num_chirp, num_Tx, num_channel, -1, 4)  # 2I + 2Q = 4   0.0s
# x_Q = readItem[:, :, :, :, :2].reshape(num_chirp, num_Tx, num_channel, -1)  # flatten the last two dims of Q data    0.002s
# x_I = readItem[:, :, :, :, 2:].reshape(num_chirp, num_Tx, num_channel, -1)  # flatten the last two dims of I data    0.002s
# data = np.array((x_Q, x_I))  # data[I/Q, Chirp, TxChannel, RxChannel, Sample]    0.001s
# data = np.transpose(data, (0, 2, 3, 1, 4))  # data[I/Q, TxChannel, RxChannel, Chirp, Sample]      0.0s
# frameWithChirp = data[0] + complex(0, 1) * data[1]  # frameWithChirp[TxChannel, RxChannel, Chirp, Sample]      0.01s
# rotation = np.array([0, -np.pi, -np.pi, 0])
# raw_mmWave = np.fft.ifft(np.transpose(np.transpose(np.fft.fft(frameWithChirp), (0, 3, 2, 1)) * np.exp(1j * rotation), (0, 3, 2, 1)))
# data = raw_mmWave.reshape(-1, num_chirp, num_sample)
# ................................................................

raw_mmWave = np.load('./realtime/Raw_data/90/mmWave.npy')
# raw_mmWave = np.load('./realtime/Alexander/data.npy')
# raw_mmWave = np.transpose(raw_mmWave, (1, 2, 0, 3))
# raw_mmWave = raw_mmWave[np.newaxis, ...]
# heatmap
center_count = 20
surround_center_count = 500

# parameters
num_chirp = 128
num_sample = 256
frames_count = 1
start_frame = 275
step_frame = 1
end_frame = start_frame + (frames_count - 1) * step_frame + 1

# file
store_path = './realtime/'
store_file = 'MyData'

# beamformed_signal_energy = np.full((center_count * surround_center_count, num_chirp, frames_count), 1)
# beamformed_signal_phase = np.full((center_count * surround_center_count, num_chirp, frames_count), 1)
# beamformed_siganl_raw = np.full((center_count * surround_center_count, num_chirp, frames_count), 0 + 2j)
pc_coordinate = []

for frame in range(start_frame, end_frame, step_frame):
    print(frame)
    data = raw_mmWave[frame]
    data = data.reshape(-1, num_chirp, num_sample)
    rangeFFTResult = np.fft.fft(data)
    # plt.figure()
    # plt.plot(abs(rangeFFTResult[0]))
    # plt.savefig('./realtime/range_test.jpg')
    # rangeFFTResult = clutter_removal(rangeFFTResult, axis=1)
    # plt.imshow(abs(rangeFFTResult[0]))
    # plt.savefig('./realtime/clutter.jpg')
    # data = np.fft.ifft(rangeFFTResult)

    # define antenna
    wavelength = 3e8 / 60e9
    spacing = wavelength / 2.0
    # vir_ant = np.array([
    #     [3, 2, 2, 3, 1, 0, 0, 1, 3, 2, 2, 3],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
    # ]).T * spacing
    vir_ant = np.array([
        [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 2, 2, 3, 1, 0, 0, 1, 3, 2, 2, 3]
    ]).T * spacing
    vir_ant[:, 2] = -vir_ant[:, 2]  # lefthand to righthand coor
    AntennaArr = model.ArrayDesign(vir_ant, name='IWR6843ISK-ODS')

    # MVDR
    # az_start, az_end = -60, 60
    # el_start, el_end = -60, 60
    az_start, az_end = 20, 141  # end not included
    el_start, el_end = -80, 71
    az_step = 1
    el_step = 1
    az_size = int((az_end - az_start) / az_step)
    el_size = int((el_end - el_start) / el_step)
    grid = estimation.FarField2DSearchGrid(start=(az_start, el_start),stop=(az_end, el_end), unit='deg', size=(az_size, el_size))
    estimator = estimation.MVDRBeamformer(AntennaArr, wavelength, grid)

    point_cloud = np.zeros((az_size, el_size, num_sample))
    for s in range(num_sample):
        temp = rangeFFTResult[..., s] @ (rangeFFTResult[..., s].conj().T) / rangeFFTResult[..., s].shape[-1]
        resv, est, point_cloud[..., s] = estimator.estimate(temp, 1, return_spectrum=True)
        # resv, est, point_cloud[..., s] = estimator.estimate(np.cov(rangeFFTResult[..., s]), 1, return_spectrum=True)
    # ax = plt.subplot(1, 1, 1)
    # doaplt.plot_spectrum(point_cloud[..., 25], grid, ax=ax)
    # ax.set_title('Beamforming')
    # plt.show()

    idx = np.argsort(point_cloud, axis=None)[::-1]
    az, el, d = np.unravel_index(idx, point_cloud.shape)
    az = az * az_step + az_start
    el = el * el_step + el_start
    # x = (d + 1) * np.cos(el * np.pi / 180.0) * np.sin(az * np.pi / 180.0)
    # y = (d + 1) * np.cos(el * np.pi / 180.0) * np.cos(az * np.pi / 180.0)
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
    matching_idx = np.where((np.abs(x) < 20) & (y > 20) & (y < 30) & (z > -10) & (z < 25))
    # matching_idx = np.where((np.abs(x) < 15) & (y > 20) & (y < 30) & (z > -3) & (z < 35))
    points = points[matching_idx]
    print(points.shape)

    # power = power[matching_idx]
    # points = points[:30000]
    # power = power[:30000]
    # print(points.shape)
    # print(power.shape)

    cluster_count = 0
    center = points[0]
    cluster = []
    while cluster_count != center_count:
        distances = np.sqrt(np.sum(np.power((points - center), 2), axis=1))
        # distances = distances[np.where(distances < 5)]
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
    print("Peforming MVDR...")

    # ROI = np.c_[
    #     np.arctan2(cluster[:,1], cluster[:,0]),
    #     np.arctan2(cluster[:,2], np.sqrt(cluster[:,0]**2 + cluster[:,1]**2)),
    #     np.sqrt(cluster[:,0]**2 + cluster[:,1]**2 + cluster[:,2]**2)
    # ]
    #
    # f = data
    # # range fft
    # f = np.fft.fft(f, axis=-1)
    # f = f.reshape(-1, num_chirp, num_sample)
    # for i, (az, el, r) in enumerate(ROI):
    #     search_direction = FarField2DSourcePlacement(np.array([az,el])[:, None].T, unit='rad')
    #     beamformed_signal = estimator.dbf(f[..., int(r)], search_direction)
    #     beamformed_signal_energy[i, ..., (frame - start_frame) // step_frame] = np.log10(np.abs(beamformed_signal) + 1)
    #     beamformed_signal_phase[i,..., (frame - start_frame) // step_frame] = np.unwrap(np.angle(beamformed_signal))

# print("computing energy and phase")
# energy = beamformed_signal_energy
# phase = beamformed_signal_phase
# print(energy.shape)
# print(phase.shape)
print("Saving it to " + store_path + store_file + ".dbf/")
if not os.path.exists(store_path + store_file + ".dbf"):
    os.mkdir(store_path + store_file + ".dbf")
np.save(store_path + store_file + ".dbf/" + store_file + ".dbf.coordinate", np.array(pc_coordinate))
# np.save(store_path + store_file + ".dbf/" + store_file + ".dbf.energy", energy)
# np.save(store_path + store_file + ".dbf/" + store_file + ".dbf.phase", phase)
# np.save(store_path + store_file + ".dbf/" + store_file + ".dbf.raw", beamformed_siganl_raw)
