import numpy as np
import matplotlib.pyplot as plt

# raw_depth = np.load('./realtime/Raw_data/90/Depth.npy')
# # data = np.load("realtime/-15--10.dbf/-15--10.dbf.coordinate.npy",mmap_mode ='r+')
# # data = np.load("realtime/MyData.dbf/Mydata.dbf.coordinate.npy",mmap_mode ='r+')
#
# frames_count = len(raw_depth)
# start_frame = 50
# step_frame = 100
# for frame in range(start_frame, frames_count):
#     print('frame: {}'.format(frame))
#     depth = raw_depth[frame]
#     fig = plt.figure(dpi=500)
#
#     ax = plt.subplot(1, 1, 1)
#     plt.imshow(depth, cmap='gray', interpolation='bicubic')
#     ax.set_title('frame: ' + str(frame))
#     plt.savefig('./realtime/images/frame{}.jpg'.format(frame))
#     # plt.show()

folder_list = ['10', '36', '36', '40', '40', '43', '47', '54', '57', '64']
image_list = [419, 202, 336, 64, 252, 85, 318, 217, 210, 402]
depth_data = []
mmWave_data = []
for index in range(len(folder_list)):
    raw_depth = np.load('./raw_data/' + folder_list[index] + '/Kinect/Depth.npy')
    raw_mmWave = np.load('./raw_data/' + folder_list[index] + '/mmWave/mmWave.npy')
    depth_data.append(raw_depth[image_list[index]])
    mmWave_data.append(raw_mmWave[image_list[index]])
np.save('depth.npy', depth_data)
np.save('mmWave.npy', mmWave_data)