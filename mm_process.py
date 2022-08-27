import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.torch_utils import time_sync

sample_num = 256  # Sample Length
chirp_num = 128  # Chirp Total
Txchannel_num = 3  # Transit Channel Total
Rxchannel_num = 4  # Receive Channel Total
LightVelocity = 3e8  # Speed of Light
FreqStart = 60e9  # Start Frequency
NumRangeFFT = 256  # Range FFT Length
slope = 30.018e12  # Frequency slope
SampleRate = 2000e3  # Sample rate
padding_azimuth = 64  # the length of the padding dimension for azimuth direction
padding_elevation = 64  # the length of the padding dimension for elevation direction
Bandwidth = slope * sample_num / SampleRate  # Sweep Bandwidth
WaveLength = LightVelocity / FreqStart  # Wave length

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


def rangeFFT(reshapedData, sample_num):
    windowedBins1D = reshapedData * np.hamming(sample_num)
    rangeFFTResult = np.fft.fft(windowedBins1D)
    return rangeFFTResult


def dopplerFFT(rangeFFTResult, chirp_num):
    windowedBins2D = rangeFFTResult * np.reshape(np.hamming(chirp_num), (1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult

def angleFFT(AOAInput, padding_azimuth, padding_elevation, numRxAntennas):
    azimuth_ant_padded = np.zeros(shape=(padding_azimuth, len(AOAInput[0]), len(AOAInput[0][0])), dtype=np.complex_)
    azimuth_ant_padded[:numRxAntennas, :, :] = AOAInput[[11, 10, 7, 6], :, :]
    elevation_ant_padded = np.zeros(shape=(padding_elevation, len(AOAInput[0]), len(AOAInput[0][0])), dtype=np.complex_)
    elevation_ant_padded[:numRxAntennas, :, :] = AOAInput[[11, 8, 3, 0], :, :]
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    azimuth_max = np.argmax(abs(azimuth_fft), axis=0)
    azimuth_max[azimuth_max > (padding_azimuth // 2) - 1] = azimuth_max[azimuth_max > (
                padding_azimuth // 2) - 1] - padding_azimuth
    wx = 2 * np.pi / padding_azimuth * azimuth_max
    x_vector = wx / np.pi
    elevation_fft = np.fft.fft(elevation_ant_padded, axis=0)
    elevation_max = np.argmax(abs(elevation_fft), axis=0)
    elevation_max[elevation_max > (padding_elevation // 2) - 1] = elevation_max[elevation_max > (
                padding_elevation // 2) - 1] - padding_elevation
    wz = 2 * np.pi / padding_elevation * elevation_max
    z_vector = wz / np.pi
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector = ypossible
    x_vector[ypossible < 0] = 0
    z_vector[ypossible < 0] = 0
    y_vector[ypossible < 0] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector

def process(readItem):
    readItem = readItem.reshape(chirp_num, Txchannel_num, Rxchannel_num, -1, 4)  # 2I + 2Q = 4   0.0s
    x_Q = readItem[:, :, :, :, :2].reshape(chirp_num, Txchannel_num, Rxchannel_num, -1)  # flatten the last two dims of Q data    0.002s
    x_I = readItem[:, :, :, :, 2:].reshape(chirp_num, Txchannel_num, Rxchannel_num, -1)  # flatten the last two dims of I data    0.002s
    data = np.array((x_Q, x_I))  # data[I/Q, Chirp, TxChannel, RxChannel, Sample]    0.001s
    data = np.transpose(data, (0, 2, 3, 1, 4))  # data[I/Q, TxChannel, RxChannel, Chirp, Sample]      0.0s
    frameWithChirp = data[0] + complex(0, 1) * data[1]  # frameWithChirp[TxChannel, RxChannel, Chirp, Sample]      0.01s
    # frameWithChirp = np.flip(frameWithChirp, 3)  # 0.0s
    rotation = np.array([0, -np.pi, -np.pi, 0])
    frameWithChirp = np.fft.ifft(np.transpose(np.transpose(np.fft.fft(frameWithChirp), (0, 3, 2, 1)) * np.exp(1j * rotation), (0, 3, 2, 1)))
    # get 1D range profile->rangeFFT
    rangeFFTResult = rangeFFT(frameWithChirp, sample_num)  # 0.01s
    plt.figure()
    plt.imshow(abs(rangeFFTResult[0][0]))
    plt.savefig('./realtime/range.jpg')
    rangeFFTResult = clutter_removal(rangeFFTResult, axis=2)  # 0.005s
    plt.imshow(abs(rangeFFTResult[0][0]))
    plt.savefig('./realtime/clutter.jpg')
    # get 2D range-velocity profile->dopplerFFT
    dopplerFFTResult = dopplerFFT(rangeFFTResult, chirp_num)  # 0.02s
    # get 2D range-angle profile->angleFFT
    dopplerFFTResult = dopplerFFTResult[:, :, 40:89, :50]  # eliminate signal with high velocity and distance  0.0s
    AOAInput = dopplerFFTResult.reshape(-1, len(dopplerFFTResult[0][0]), len(dopplerFFTResult[0][0][0]))   # 0.0s
    x_vector, y_vector, z_vector = angleFFT(AOAInput, padding_azimuth, padding_elevation, Rxchannel_num)  # 0.02s

    # 0.02s
    size = 32
    image = np.zeros((size, size)).astype(float)
    resolution = 0.133
    for ii in range(len(x_vector)):
        for jj in range(len(x_vector[0])):
            R = jj * LightVelocity / (2 * Bandwidth)
            x = int(x_vector[ii, jj] * R // resolution) + size // 2
            z = int(z_vector[ii, jj] * R // resolution) + size // 2
            if (x >= size) | (x < 0) | (z >= size) | (z < 0):
                continue
            if ((y_vector[ii, jj] * R < image[z, x]) | (image[z, x] == 0)) & (y_vector[ii, jj] * R > 0.0):
                image[z, x] = y_vector[ii, jj] * R
    max_value = np.ceil(image.max())
    image = image / max_value * 256
    image = cv2.resize(image, (416, 416))

    return image