# .............................tianyu point cloud.......................................
import numpy as np
import plotly.graph_objects as go
# data = np.load("realtime/-15--10.dbf/-15--10.dbf.coordinate.npy",mmap_mode ='r+')
data = np.load("realtime/MyData.dbf/Mydata.dbf.coordinate.npy",mmap_mode ='r+')
frame = 0
fig = go.Figure(data=[go.Scatter3d(x=data[frame][:,0], y=data[frame][:,1], z=data[frame][:,2],mode='markers',
    marker=dict(
        size=12,
        color=data[frame][:,3],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
        ))])
fig.update_layout(autosize=False,
    width=1500,
    height=1500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))
fig.show()
# .............................tianyu point cloud.......................................

# .............................My point cloud matplotlib.......................................
# import numpy as np
# import matplotlib.pyplot as plt
# # data = np.load("realtime/-15--10.dbf/-15--10.dbf.coordinate.npy",mmap_mode ='r+')
# data = np.load("realtime/MyData.dbf/Mydata.dbf.coordinate.npy",mmap_mode ='r+')
# frame = 0
# point_cloud = data[frame]
# fig = plt.figure(dpi=500)
# ax = fig.add_subplot(111, projection='3d')
#
# colors = point_cloud[:, 3]
#
# ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
#            cmap='Spectral',
#            c=colors,
#            s=12,
#            linewidth=0,
#            alpha=1,
#            marker=".")
#
# plt.title('Point Cloud')
# # ax.axis('scaled')  # {equal, scaled}
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.savefig('./realtime/point.jpg')
# plt.show()
# .............................My point cloud matplotlib.......................................

# .............................My point cloud o3d.......................................
# import open3d as o3d
# import numpy as np
# # data = np.load("realtime/-15--10.dbf/-15--10.dbf.coordinate.npy",mmap_mode ='r+')
# data = np.load("realtime/MyData.dbf/Mydata.dbf.coordinate.npy",mmap_mode ='r+')
# frame = 0
# point_cloud = data[frame, :, :3]
# ptCloud = o3d.geometry.PointCloud()
# ptCloud.points = o3d.utility.Vector3dVector(point_cloud)
# o3d.visualization.draw_geometries([ptCloud], window_name="GT")
# .............................My point cloud o3d.......................................

# .............................My 2d.......................................
# import numpy as np
# import matplotlib.pyplot as plt
#
# raw_depth = np.load('./realtime/Raw_data/90/Depth.npy')
# # data = np.load("realtime/-15--10.dbf/-15--10.dbf.coordinate.npy",mmap_mode ='r+')
# data = np.load("realtime/MyData.dbf/Mydata.dbf.coordinate.npy",mmap_mode ='r+')
#
# # frames_count = len(raw_depth)
# frames_count = 10
# start_frame = 40
# step_frame = 40
# for frame in range(frames_count):
#     print(start_frame + frame * step_frame)
#     depth = raw_depth[start_frame + frame * step_frame]
#     # depth = raw_depth[frame]
#     point_cloud = data[frame]
#     ax1 = plt.subplot(1, 2, 1)
#     # plt.xlim((-50, 50))
#     # plt.ylim((-20, 50))
#     colors = point_cloud[:, 3]
#
#     ax1.scatter(point_cloud[:, 0], point_cloud[:, 2],
#                cmap='jet',
#                c=colors,
#                s=12,
#                linewidth=0,
#                alpha=1,
#                marker=".")
#
#     ax1.set_title('2Dscatter')
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Z')
#
#     ax2 = plt.subplot(1, 2, 2)
#     plt.imshow(depth, cmap='gray', interpolation='bicubic')
#     ax2.set_title('Depth')
#     plt.show()
# .............................My 2d.......................................

# .............................Tianyu 2d.......................................
# import numpy as np
# import plotly.graph_objects as go
# # data = np.load("realtime/-15--10.dbf/-15--10.dbf.coordinate.npy",mmap_mode ='r+')
# data = np.load("realtime/MyData.dbf/Mydata.dbf.coordinate.npy",mmap_mode ='r+')
# frame = 2
# fig = go.Figure(data=[go.Scatter(x=data[frame][:,0], y=data[frame][:,2],mode='markers',
#     marker=dict(
#         size=12,
#         color=data[frame][:,3],                # set color to an array/list of desired values
#         colorscale='Viridis',   # choose a colorscale
#         opacity=0.8
#         ))])
# fig.update_layout(autosize=False,
#     width=1500,
#     height=1500,
#     margin=dict(
#         l=50,
#         r=50,
#         b=100,
#         t=100,
#         pad=4
#     ))
# fig.show()
# .............................Tianyu 2d.......................................

import numpy as np
import matplotlib.pyplot as plt

typ = np.dtype((np.uint16, (424, 512)))
# raw_depth = np.fromfile('./realtime/Raw_data/right/Depth.npy', dtype=typ)
raw_depth = np.load('./realtime/Raw_data/90/Depth.npy')
depth = raw_depth[275]
# data = np.load("realtime/-15--10.dbf/-15--10.dbf.coordinate.npy",mmap_mode ='r+')
data = np.load("realtime/MyData.dbf/Mydata.dbf.coordinate.npy",mmap_mode ='r+')

# frames_count = len(raw_depth)
frames_count = 1
start_frame = 275
for frame in range(frames_count):
    print(start_frame + frame)
    point_cloud = data[frame]
    ax1 = plt.subplot(1, 2, 1)
    # plt.xlim((-50, 50))
    # plt.ylim((-20, 50))
    colors = point_cloud[:, 3]

    ax1.scatter(point_cloud[:, 0], point_cloud[:, 2],
               cmap='jet',
               c=colors,
               s=12,
               linewidth=0,
               alpha=1,
               marker=".")

    ax1.set_title('2Dscatter')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')

    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap='gray', interpolation='bicubic')
    ax2.set_title('Depth')
    plt.show()