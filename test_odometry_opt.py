import argparse
import os
import time
import torch.utils.data
import cv2
import numpy as np
from liegroups.numpy.se3 import SE3Matrix
import torch.nn as nn
import ceres_python
import matplotlib.pyplot as plt

from data.odom_loader import KittiOdomLoader
from models.nets import RGBNet
import models.se3 as se3
import utils

parser = argparse.ArgumentParser(description='Training the network')
parser.add_argument('--root_dir', type=str, default='/home/song/Documents/data/kitti_odom',
                    help='path to the dataset folder./data/szb/kitti_depth /home/song/Documents/data/kitti_depth'
                    )
parser.add_argument('--test_data_dir', type=str, default='data/filenames/10.txt',
                    help='path to the test dataset file.'
                    )
parser.add_argument('--output_directory', type=str, default='results',
                    help='where save dispairities for tested images'
                    )
parser.add_argument('--resume',type=str,
                    default='results/featnet_model.pth',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--image_size', type=tuple, default=(352, 1216),
                    help='image size in the network, as type tuple (height, width)'
                    )
parser.add_argument('--max_iter',
                    default=3, type=int,
                    help='The maximum number of iterations at each pyramids.\n')
parser.add_argument('--point_size', default=1024 * 8, type=int, help='input number of points\n')
args = parser.parse_args()


def mae_loss(pred_pose, gt_pose):
    xi_est = se3.se3_log(pred_pose)
    inv_pose_est = se3.se3_exp(-xi_est)

    d_pose = torch.bmm(inv_pose_est, gt_pose)
    d_xi = se3.se3_log(d_pose)
    angle_error = torch.sqrt((d_xi[0, 3:] ** 2).sum())
    trans_error = torch.sqrt((d_xi[0, :3] ** 2).sum())
    return angle_error, trans_error

cuda = torch.cuda.is_available()

if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}' ... ".format(args.resume),
          end='')
    checkpoint = torch.load(args.resume, map_location=device)
else:
    print("No checkpoint found at '{}'".format(args.resume))

model = RGBNet().to(device)

cmap = plt.cm.jet

if checkpoint is not None:
    model.load_state_dict(checkpoint)
    print("=> checkpoint state loaded.")
model.eval()

model = torch.nn.DataParallel(model)


# Data loading code
print("=> creating data loaders ... ")

dataset = KittiOdomLoader(args.root_dir, args.test_data_dir, 'test', args.image_size, args.point_size)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=2,
                                     pin_memory=True,)

# main loop
print("=> starting main loop ...")


count = 0
ang_err = 0.
trans_err = 0.
show_images = False
acc_pose = np.eye(4)
poses = list()
for i, batch_data in enumerate(loader):
    start = time.time()
    batch_data = utils.to_device(batch_data, device)
    # we use log depth to represent the depth values
    ref_image = batch_data['second_im']
    tar_image = batch_data['first_im']
    ref_ptc = batch_data['second_pts']
    cam = batch_data['camera']
    gt_pose = batch_data['pose']
    b, _, n = ref_ptc.shape
    K = torch.eye(3, dtype=cam.dtype, device=cam.device).view(1, 3, 3).repeat(b, 1, 1)
    K[:, 0, 0] = cam[:, 0]
    K[:, 1, 1] = cam[:, 1]
    K[:, 0, 2] = cam[:, 2]
    K[:, 1, 2] = cam[:, 3]
    pix = torch.bmm(K, ref_ptc)
    pix[:, 0, :] = pix[:, 0, :] / pix[:, 2, :]
    pix[:, 1, :] = pix[:, 1, :] / pix[:, 2, :]
    ref_ind = pix[:, :2, :]
    data_time = time.time() - start

    start = time.time()
    with torch.no_grad():
        ref_feature = model(ref_image)
        tar_feature = model(tar_image)

    pose_xi = np.zeros((1, 6))
    K = K.squeeze().data.cpu().numpy()
    ref_ptc = ref_ptc.squeeze().data.cpu().numpy()
    for scale in range(len(ref_feature)):
        K_pyr = K / (1 << (3 - scale))
        K_pyr[2, 2] = 1.0
        w = args.image_size[1] / (1 << (3 - scale))
        h = args.image_size[0] / (1 << (3 - scale))
        ref_ind_pyr = ref_ind / (1 << (3 - scale))
        ref_sample_x = 2.0 * ref_ind_pyr[:, 0:1, :] / max(w - 1, 1) - 1.0
        ref_sample_y = 2.0 * ref_ind_pyr[:, 1:2, :] / max(h - 1, 1) - 1.0
        ref_sample = torch.cat((ref_sample_x, ref_sample_y), dim=1).unsqueeze(2).permute(0, 3, 2, 1)  # B, N, 1, 2
        '''
               /*
                * part1. observation:
                * kNumObservations: number of observations
                * tar_image: target image for projection in shape [C, H, W]
                * ref_pt3: reference 3D space point in shape [3, n]
                * ref_intensity: reference intensity corresponding to ref_pt3 in shape [C, n]
                *
                * part2. initial estimation:
                * init_pose: initial pose in shape [1, 6]
                *
                * part3. system setting
                * cam_k: camera intrinsic parameters [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                * max_iter: max iteration number
                */
                '''
        ref_pts_feature = nn.functional.grid_sample(ref_feature[scale], ref_sample).squeeze().data.cpu().numpy()  # C, N
        tar_map_feature = tar_feature[scale].squeeze().data.cpu().numpy()  # C, H, W

        # iter_time = time.time()
        pose_xi = ceres_python.opt_photometric(args.point_size, tar_map_feature, ref_ptc, ref_pts_feature,
                                               pose_xi, K_pyr, 3)

    curr_pose = SE3Matrix.exp(pose_xi).as_matrix()
    count += 1
    acc_pose = acc_pose.dot(curr_pose)
    poses.append(acc_pose)
    a_e, t_e = mae_loss(torch.from_numpy(curr_pose.astype(np.float32)).cuda(), gt_pose)
    ang_err += a_e
    trans_err += t_e
    if show_images:
        x = ref_feature[-1].mean(1, True)
        fea = np.squeeze(x.data.cpu().numpy())
        fea = (fea - np.min(fea)) / (np.max(fea) - np.min(fea))
        fea = 255 * cmap(fea)[:, :, :3]  # H, W, C
        rgb = 255 * np.squeeze(batch_data['second_im'].transpose(2, 3).data.cpu().numpy()).transpose()
        cv2.imshow('rgb', rgb.astype('uint8'))
        cv2.imshow('feature', fea.astype('uint8'))
        cv2.waitKey(5)

    gpu_time = time.time() - start
    print(len(poses), ',', ang_err/count, ',', trans_err/count, 'time: ', gpu_time, 's')

print(ang_err/count, trans_err/count)
seq = args.test_data_dir.split('/')[-1][:2]
with open('results/tanh_rgbnet/' + seq + '_ceres.txt', "w") as f:
    for x in poses:
        for r in range(3):
            for c in range(4):
                f.write("%f " % x[r, c])
        f.write("\n")

