from tkinter import CENTER
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
import os
import doatools.doatools.model as model
import doatools.doatools.plotting as doaplt
from doatools.doatools.plotting import plot_array, plot_spectrum
import doatools.doatools.estimation as estimation
from doatools.doatools.estimation.core import SpectrumBasedEstimatorBase, ensure_covariance_size
from doatools.doatools.model.sources import FarField2DSourcePlacement
# import plotly.graph_objects as go
from scipy.io import savemat
# import cupy as cp

# from raw_data_extract import *

def IWR6843TDM_lazy(dfile, num_frame, num_chirp, num_Tx, num_channel = 4, num_sample=256):
    """ Lazy load adc data

    Parameters
    ----------
    dfile : str
        an reorganized adc_data file
    num_frame : int
    num_chirp : int
    num_Tx: int
    num_channel: int
    num_sample : int

    Returns
    -------
    np.memmap
        the I, Q data of each sample are not reordered
    """
    if num_frame == -1:
        num_frame = os.path.getsize(dfile) // num_chirp // num_Tx // num_channel // num_sample // 2 // 2
    return np.memmap(dfile, mode='r', dtype=np.int16, shape=(num_frame, num_chirp, num_Tx, num_channel, num_sample * 2))


def reorder_IQ(d):
    """ reorder the IIQQ format of data to I+1j*Q, the returned data is in memory

    Parameters
    ----------
    d : np.memmap
        the data loaded by awr1642_lazy or IWR6843TDM_lazy

    Returns
    -------
    ndarray
    """
    shape = list(d.shape)
    shape[-1] = -1
    shape.append(4)
    d = d.reshape(shape)
    d = d[..., :2] + 1j * d[..., 2:]
    d = d.reshape(shape[:-1])
    return d

def phase_flip(d):
    """
    add -180 degree phase inversion to the upside-down Rx on IWR6843ISK-ODS

    Parameters
    ----------
    d : np.ndarray
        the reordered IQ data

    Returns
    -------
    ndarray

    """
    rotation = np.array([0, -np.pi, -np.pi, 0, 0, -np.pi, -np.pi, 0, 0, -np.pi, -np.pi, 0]).reshape(3, 4)
    # rotation = -rotation
    # rotation = rotation + np.pi
    # rotation = -(rotation + np.pi)
    return np.fft.ifft(np.fft.fft(d, axis=-1) * np.exp(1j*rotation)[...,None])  # broardcase: (..., Tx, Channel, Sample) * (3, 4, None)


wavelength = 3e8 / 60e9
spacing = wavelength / 2.0
vir_ant = np.array([
    [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 2, 3, 3, 2, 2, 3, 1, 0, 0, 1]
]).T * spacing
vir_ant[:, 2] = -vir_ant[:, 2]  # lefthand to righthand coor
AntennaArr = model.ArrayDesign(vir_ant, name='IWR6843ISK-ODS')


class MVDR(SpectrumBasedEstimatorBase):
    def __init__(self, array, wavelength, search_grid, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)

    @staticmethod
    def _cov(signal):
        """
        get signal's covariance matrix

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time

        Returns
        -------
        np.ndarray
            covariance matrix
        """
        return signal @ (signal.conj().T) / signal.shape[-1]

    def estimate(self, signal):
        """
        estimate the MVDR spectrum

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time

        Returns
        -------
        sp: np.ndarray
            MDDR spectrum. The size and resolution is determined by search_grid
        """
        R = MVDR._cov(signal)
        ensure_covariance_size(R, self._array)

        sp = estimation.beamforming.f_mvdr(self._get_atom_matrix(), R)
        sp = sp.reshape(self._search_grid.shape)
        return sp

    def dbf(self, signal, source):
        """
        digital beamforming using MVDR

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time
        source: doatools.model.sources.SourcePlacement
            sources where MVDR perform dbf

        Returns
        -------
        np.ndarray
            beamformed signal, shape=(*grid.shape, t)
        """
        R = MVDR._cov(signal)
        ensure_covariance_size(R, self._array)

        sv = self._array.steering_matrix(
            source, self._wavelength,
            perturbations='known'
        )
        R_invSV = np.linalg.lstsq(R, sv, None)[0]
        spatial_filter = R_invSV / (sv.conj() * R_invSV)
        spatial_filter = spatial_filter.conj()  # channel, source
        ret = np.einsum("ct, cs->st", signal, spatial_filter)  # mult each channel with the weight and sum up, here t is slow time
        return ret

    def dbf_grid(self, signal, grid=None):
        """
        grid digital beamforming using MVDR

        Parameters
        ----------
        signal: np.ndarray
            c by t matrix where c is the channel and t is the (slow) time
        grid: doatools.estimation.grid.SearchGrid
            a search grid where MVDR perform dbf, the default DoA grid is used if grid is None

        Returns
        -------
        np.ndarray
            beamformed signal, shape=(*grid.shape, t)
        """

        if grid == None:
            grid = self._search_grid
        ret = self.dbf(signal, grid.source_placement)
        ret = ret.reshape(*grid.shape, ret.shape[-1])  # source placement is flattened, so reshape is required
        return ret

def cloudpoint(dfile, num_frame, num_chirp, num_Tx, num_channel, num_sample, ob_frame, az_start, az_end, az_step, el_start, el_end, el_step):

    data = IWR6843TDM_lazy(dfile=dfile, num_frame=num_frame, num_chirp=num_chirp, num_Tx=num_Tx, num_channel=num_channel, num_sample=num_sample)
    data = data[ob_frame]
    data = reorder_IQ(data)
    data = phase_flip(data)

    # range fft
    data = np.fft.fft(data, axis=-1)
    # plt.figure()
    # plt.plot(abs(data[:, 0, 0, :]))
    # plt.savefig('./realtime/range.jpg')
    data = data.reshape(num_chirp, num_Tx*num_channel, num_sample)
    # print(data.shape)

    # clutter removal: in static scenario do not enable it
    # data = data - data.mean(axis=0)

    az_size = int((az_end-az_start)/az_step)
    el_size = int((el_end-el_start)/el_step)
    grid = estimation.FarField2DSearchGrid(start=(az_start,el_start),stop=(az_end,el_end), unit='deg', size=(az_size,el_size))
    # estimator = estimation.beamforming.MVDRBeamformer(AntennaArr, wavelength, grid)
    estimator = MVDR(AntennaArr, wavelength, grid)
    # estimator = estimation.music.MUSIC(AntennaArr, wavelength, grid)


    point_cloud = np.zeros((az_size, el_size, num_sample))
    for s in range(num_sample):
        point_cloud[..., s] = estimator.estimate(data[..., s].T)
    # ax = plt.subplot(1, 1, 1)
    # doaplt.plot_spectrum(point_cloud[..., 1], grid, ax=ax)
    # ax.set_title('Beamforming')
    # plt.show()
    return estimator, point_cloud

dfile = [
        # "./realtime/alexander/adc_data_Raw.bin",
        # "./realtime/marcus/adc_data_Raw.bin",
#         "./realtime/adc_data_Raw_0.bin",
#         "./realtime/adc_data_Raw_1.bin",
#         "./realtime/adc_data_Raw_2.bin",
        "./realtime/Raw_data/test/adc_data_Raw_0.bin",
#         "/mnt/stuff/mmFace_data/mmwave/tianyu/adc_data_Raw.bin",
#         "/mnt/stuff/mmFace_data/mmwave/jackson/adc_data_Raw.bin",
#         "/mnt/stuff/mmFace_data/mmwave/chenhan/adc_data_Raw.bin",
#         "/mnt/stuff/mmFace_data/mmwave/aditya/adc_data_Raw.bin"
        ]
#store_file = ["alexander","marcus","christian","jieyi","rocky","roger","tianyu","jackson","chenhan","aditya"]
# dfile = ["./realtime/adc_data_Raw_0.bin",
#          "./realtime/adc_data_Raw_1.bin",
#          "./realtime/adc_data_Raw_2.bin"]
store_file = ["./MyData"]

store_path ="realtime/"
for file_idx, df in enumerate(dfile):
    print('loading files {}'.format(df))
    # az_start, az_end = -60, 60
    az_start, az_end =  20, 141 # end not included
    az_step = 1
    # el_start, el_end = -60, 60
    el_start, el_end = -80, 71
    el_step = 1
    num_frame=-1
    num_chirp=128
    num_Tx=3
    num_channel=4
    num_sample=256
    frames_count = 1
    #start frame
    start_frame = 20

    #heatmap
    center_count = 10
    surround_center_count = 4000

    mmWavedata = IWR6843TDM_lazy(dfile=df, num_frame=start_frame+frames_count, num_chirp=num_chirp, num_Tx=num_Tx, num_channel=num_channel, num_sample=num_sample)
    beamformed_signal_energy = np.full((center_count*surround_center_count, num_chirp, frames_count),1)
    beamformed_signal_phase = np.full((center_count*surround_center_count, num_chirp, frames_count),1)

    beamformed_siganl_raw = np.full((center_count*surround_center_count, num_chirp, frames_count), 0+2j)

    pc_coordinate = []
    print(mmWavedata.shape)


    #for frame in range(mmWavedata.shape[0]):
    for frame in range(start_frame,start_frame+frames_count):
        ob_frame=frame
        est, p = cloudpoint(dfile=df, num_frame=num_frame, num_chirp=num_chirp, num_Tx=num_Tx, num_channel=num_channel, num_sample=num_sample, ob_frame=ob_frame, az_start=az_start, az_end=az_end, az_step=az_step, el_start=el_start, el_end=el_end, el_step=el_step)
        idx = np.argsort(p, axis=None)[::-1]
        az, el, d = np.unravel_index(idx, p.shape)
        az = az*az_step + az_start
        el = el*el_step + el_start
        z = d * np.sin(el*np.pi/180.0)
        x = d * np.cos(el*np.pi/180.0) * np.cos(az*np.pi/180.0)
        y = d * np.cos(el*np.pi/180.0) * np.sin(az*np.pi/180.0)
        print("frame: #%d"%(frame - start_frame))
        power = np.log(p[((az-az_start)/az_step).astype(int), ((el-el_start)/el_step).astype(int), d])
        power_idx = np.argsort(power)[::-1]
        x = x.reshape((-1,1))
        y = y.reshape((-1,1))
        z = z.reshape((-1,1))
        points = np.concatenate((x,y,z),axis=1)

        power = power[power_idx]
        points = points[power_idx]
        y = points[:,1]
        z = points[:,2]
        x = points[:,0]
        matching_idx = np.where((np.abs(x)<15)&(y>10) & (y<20) & (z>-3) & (z<35))
        # matching_idx = np.where((np.abs(x) < 40) & (y > 10) & (y < 20) & (z > -20) & (z < 55))
        points = points[matching_idx]
        print(points.shape)
        cluster_count = 0
        center = points[0]
        cluster = []

        while cluster_count != center_count:
            distances = np.sqrt(np.sum(np.power((points - center),2)*np.array([1,4,1]),axis=1))
            distances_idx = np.argsort(distances,axis=None)[:surround_center_count]

            temp_cluster = points[distances_idx]
            temp_cluster_power = power[distances_idx].reshape((-1,1))
            temp_cluster = np.concatenate((temp_cluster,temp_cluster_power),axis=1)

            indices = np.setxor1d(np.arange(points.shape[0]),distances_idx)
            points = points[indices]
            power = power[indices]

            power_idx = np.argsort(power)[::-1]
            points = points[power_idx]
            power = power[power_idx]
            center = points[0]

            cluster.append(temp_cluster)
            cluster_count +=1
            print("cluster:%d"%cluster_count)
        cluster = np.concatenate(cluster,axis=0)
        pc_coordinate.append(cluster)
        print("Peforming MVDR...")
#        cluster = []
#        centers = points[matching_idx][:center_count]
#        for center in centers:
#            distances = np.sqrt(np.sum(np.power((points - center),2),axis=1))
#            distances_idx = np.argsort(distances,axis=None)[:surround_center_count]
#
#            temp_cluster = points[distances_idx]
#            temp_cluster_power = power[distances_idx].reshape((-1,1))
#            temp_cluster = np.concatenate((temp_cluster,temp_cluster_power),axis=1)
#
#            indices = np.setxor1d(np.arange(points.shape[0]),distances_idx)
#            points = points[indices]
#            power = power[indices]
#
#            cluster.append(temp_cluster)
#        cluster = np.concatenate(cluster,axis=0)
        ROI = np.c_[
            np.arctan2(cluster[:,1],cluster[:,0]),
            np.arctan2(cluster[:,2],np.sqrt(cluster[:,0]**2 + cluster[:,1]**2)),
            np.sqrt(cluster[:,0]**2 + cluster[:,1]**2 + cluster[:,2]**2)
        ]

        f = mmWavedata[frame]
        f = reorder_IQ(f)
        f = phase_flip(f)
        # range fft
        f = np.fft.fft(f, axis=-1)
        f = f.reshape(num_chirp, num_Tx*num_channel, num_sample)
        for i, (az, el, r) in enumerate(ROI):
            search_direction = FarField2DSourcePlacement(np.array([az,el])[:, None].T, unit='rad')
            beamformed_signal = est.dbf(f[..., int(r)].T, search_direction)
            beamformed_signal_energy[i, ..., frame-start_frame] = np.log10(np.abs(beamformed_signal)+1)
            beamformed_signal_phase[i,...,frame-start_frame] = np.unwrap(np.angle(beamformed_signal))

    print("computing energy and phase")
    energy = beamformed_signal_energy
    phase = beamformed_signal_phase
    print(energy.shape)
    print(phase.shape)
    print("Saving it to "+ store_path +store_file[file_idx]+".dbf/")
    if not os.path.exists(store_path +store_file[file_idx]+".dbf"):
        os.mkdir(store_path +store_file[file_idx]+".dbf")
    np.save(store_path + store_file[file_idx]+".dbf/" +store_file[file_idx]+".dbf.coordinate",np.array(pc_coordinate))
    np.save(store_path + store_file[file_idx]+".dbf/" +store_file[file_idx]+".dbf.energy",energy)
    np.save(store_path + store_file[file_idx]+".dbf/" +store_file[file_idx]+".dbf.phase",phase)
    np.save(store_path + store_file[file_idx]+".dbf/" +store_file[file_idx]+".dbf.raw",beamformed_siganl_raw)


