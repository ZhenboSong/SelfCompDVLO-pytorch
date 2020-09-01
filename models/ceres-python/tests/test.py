import ceres_python
import numpy as np
import time
pt2 = np.loadtxt("/data/songzb/projects/patch-matching/models/ceres-python/data/p2d.txt").transpose().astype(np.float32)
pt3 = np.loadtxt("/data/songzb/projects/patch-matching/models/ceres-python/data/p3d.txt").transpose().astype(np.float32)
init_pose = np.zeros((1, 6))
fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7
cam_k = np.eye(3)
cam_k[0, 0] = fx
cam_k[0, 2] = cx
cam_k[1, 1] = fy
cam_k[1, 2] = cy
end_time = time.time()
res_pose = ceres_python.opt_reproject(pt2.shape[1], pt3, pt2, init_pose, cam_k)
print(res_pose)
print("time:", time.time() - end_time, "s")


