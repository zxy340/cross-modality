import numpy as np
import os
import cv2
from doatools.doatools import model, estimation
import matplotlib.pyplot as plt
import multiprocessing

def data_process(dataset_list, process_index):
    start_list = process_index * 10
    end_list = process_index * 10 + 11
    for current_dataset_index in range(start_list, end_list):
        print('Current processed dataset is {}'.format(dataset_list[current_dataset_index]))
        load_path = '/home/xiaoyu/MyProject/data/raw_data/'
        store_path = '/home/xiaoyu/MyProject/datasets/new_mmWave_Chinese/images/'
        if not os.path.exists(load_path + dataset_list[current_dataset_index]):
            print('No {} folder data'.format(dataset_list[current_dataset_index]))
            continue

        raw_mmWave = np.load(load_path + dataset_list[current_dataset_index] + '/mmWave/mmWave.npy')
        # parameters
        num_chirp = 128
        num_sample = 256
        frames_count = len(raw_mmWave)

        for frame in range(frames_count):
            data = raw_mmWave[frame]
            data = data.reshape(-1, num_chirp, num_sample)
            rangeFFTResult = np.fft.fft(data)

            # define antenna
            wavelength = 3e8 / 60e9
            spacing = wavelength / 2.0
            vir_ant = np.array([
                [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 2, 2, 3, 1, 0, 0, 1, 3, 2, 2, 3]
            ]).T * spacing
            vir_ant[:, 2] = -vir_ant[:, 2]  # lefthand to righthand coor
            AntennaArr = model.ArrayDesign(vir_ant, name='IWR6843ISK-ODS')

            # MVDR
            az_start, az_end = 20, 141
            el_start, el_end = -80, 71
            az_step = 1
            el_step = 1
            az_size = int((az_end - az_start) / az_step)
            el_size = int((el_end - el_start) / el_step)
            grid = estimation.FarField2DSearchGrid(start=(az_start, el_start), stop=(az_end, el_end), unit='deg',
                                                   size=(az_size, el_size))
            estimator = estimation.MVDRBeamformer(AntennaArr, wavelength, grid)

            point_cloud = np.zeros((az_size, el_size, num_sample))
            for s in range(num_sample):
                temp = rangeFFTResult[..., s] @ (rangeFFTResult[..., s].conj().T) / rangeFFTResult[..., s].shape[-1]
                resv, est, point_cloud[..., s] = estimator.estimate(temp, 1, return_spectrum=True)

            idx = np.argsort(point_cloud, axis=None)[::-1]
            az, el, d = np.unravel_index(idx, point_cloud.shape)
            az = az * az_step + az_start
            el = el * el_step + el_start
            z = d * np.sin(el * np.pi / 180.0)
            x = d * np.cos(el * np.pi / 180.0) * np.cos(az * np.pi / 180.0)
            y = d * np.cos(el * np.pi / 180.0) * np.sin(az * np.pi / 180.0)

            power = np.log(
                point_cloud[((az - az_start) / az_step).astype(int), ((el - el_start) / el_step).astype(int), d])
            power_idx = np.argsort(power)[::-1]
            x = x.reshape((-1, 1))
            y = y.reshape((-1, 1))
            z = z.reshape((-1, 1))
            points = np.concatenate((x, y, z), axis=1)

            power = power[power_idx]
            points = points[power_idx]
            y = points[:, 1]
            z = points[:, 2]
            x = points[:, 0]
            matching_idx = np.where((np.abs(x) < 20) & (y > 20) & (y < 30) & (z > -10) & (z < 25))
            points = points[matching_idx]

            center_count = 20
            surround_center_count = 500
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

                indices = np.setxor1d(np.arange(points.shape[0]), distances_idx)
                points = points[indices]
                power = power[indices]

                power_idx = np.argsort(power)[::-1]
                points = points[power_idx]
                power = power[power_idx]

                center = points[0]

                cluster.append(temp_cluster)
                cluster_count += 1
            cluster = np.concatenate(cluster, axis=0)

            ax = plt.subplot(1, 1, 1)
            plt.xlim((-20, 20))
            plt.ylim((-10, 20))
            colors = cluster[:, 3]
            ax.scatter(cluster[:, 0], cluster[:, 2],
                       cmap='jet',
                       c=colors,
                       s=12,
                       linewidth=0,
                       alpha=1,
                       marker=".")
            plt.savefig('./realtime/temp' + str(process_index) + '.jpg')
            plt.close()
            image = cv2.imread('./realtime/temp' + str(process_index) + '.jpg')
            image = cv2.resize(image, (416, 416))
            if frame < 10:
                fig_name = str(dataset_list[current_dataset_index]) + '_00' + str(int(frame)) + '.jpg'
            elif frame < 100:
                fig_name = str(dataset_list[current_dataset_index]) + '_0' + str(int(frame)) + '.jpg'
            else:
                fig_name = str(dataset_list[current_dataset_index]) + '_' + str(int(frame)) + '.jpg'
            cv2.imwrite(store_path + fig_name, image)


if __name__ == "__main__":
    dataset_list = []
    for i in range(1, 101):
        dataset_list.append(str(i))

    processes = []
    for i in range(10):
        process = multiprocessing.Process(target=data_process, args=(dataset_list, i))
        processes.append(process)
        process.start()

    for proc in processes:
        proc.join()