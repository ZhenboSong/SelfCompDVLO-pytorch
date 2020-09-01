from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import data.depth_interpolate as di


class KittiOdomLoader(Dataset):
    def __init__(self, root_dir, data_dir, mode, im_size, pt_size):
        self.root_dir = root_dir
        self.im_size = im_size
        self.pt_size = pt_size
        self.mode = mode
        self.dicts = []
        if mode == 'train':
            self.im_trans = transforms.Compose(
                [transforms.Resize(im_size, interpolation=Image.NEAREST),
                 # transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                 transforms.ToTensor(),
            ])
        else:
            self.im_trans = transforms.Compose(
                [transforms.Resize(im_size, interpolation=Image.NEAREST),
                 transforms.ToTensor(),
                 ])

        self.dm_trans = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(im_size, interpolation=Image.NEAREST),
             transforms.ToTensor(),]
        )

        f = open(data_dir, "r")
        for line in f:
            line = line.strip("\n")
            fields = line.split(" ")
            self.dicts.append({"sequence_id": int(fields[0]), "image_id": int(fields[1])})
        f.close()

    def get_pose(self, idx):
        pose_dir = '%s/poses/%02d.txt' % (self.root_dir, self.dicts[idx]['sequence_id'])
        image_id = self.dicts[idx]['image_id']
        poses = []
        try:
            with open(pose_dir, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam2 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam2 = T_w_cam2.reshape(3, 4)
                    T_w_cam2 = np.vstack((T_w_cam2, [0, 0, 0, 1]))
                    poses.append(T_w_cam2)

        except FileNotFoundError:
            print('Ground truth poses are not avaialble for sequence ')
        return np.dot(np.linalg.inv(poses[image_id]), poses[image_id + 1])

    def get_image_pair(self, idx):
        image_dir_0 = '%s/sequences/%02d/image_2/%06d.png' \
                      % (self.root_dir, self.dicts[idx]['sequence_id'], self.dicts[idx]['image_id'])
        image_0 = Image.open(image_dir_0)
        width = image_0.size[0]
        height = image_0.size[1]
        delta_w = width - 1216
        delta_h = height - 352
        left_d = int(delta_w / 2)
        right_d = width - (delta_w - left_d)

        image_0 = image_0.crop([left_d, delta_h, right_d, height])
        image_dir_1 = '%s/sequences/%02d/image_2/%06d.png' \
                      % (self.root_dir, self.dicts[idx]['sequence_id'], self.dicts[idx]['image_id'] + 1)
        image_1 = Image.open(image_dir_1)
        image_1 = image_1.crop([left_d, delta_h, right_d, height])
        return image_0, image_1, [left_d, delta_h, right_d, height]

    def sample_randomly(self, pts):
        n, _ = pts.shape
        if n >= self.pt_size:
            ind = random.sample(range(n), self.pt_size)
            return pts[ind]
        else:
            ind = random.sample(range(n), self.pt_size-n)
            n_pts = pts[ind]
            return np.concatenate((pts, n_pts), axis=0)

    def get_depth_map(self, idx, calib, image_size):
        w, h = image_size
        # get first depth map
        pc_dir = '%s/sequences/%02d/velodyne/%06d.bin' \
                   % (self.root_dir, self.dicts[idx]['sequence_id'], self.dicts[idx]['image_id'])
        pc = np.fromfile(pc_dir, dtype=np.float32, count=-1).reshape([-1, 4])
        c_pc = np.dot(pc[:, 0:3], calib['Tr'][0:3, 0:3].T) + calib['Tr'][0:3, 3] + calib['d2'].T
        c_pc = c_pc[c_pc[:, 2] > 1]
        c_pc = c_pc[c_pc[:, 2] < 80]

        px = np.dot(c_pc, calib['K2'].T)
        depth = px[:, 2]
        px[:, 0] = np.round(px[:, 0] / depth)
        px[:, 1] = np.round(px[:, 1] / depth)

        ind = (px[:, 0] < w) & (px[:, 0] >= 0) &\
              (px[:, 1] < h) & (px[:, 0] >= 0)
        c_pc = c_pc[ind]
        depth = depth[ind]
        px = px[ind]

        trail = np.argsort(depth, -1, 'quicksort')[::-1]
        depth = depth[trail]
        c_pc = c_pc[trail]
        px = px[trail, :]
        depthmap0 = np.zeros(image_size)
        depthmap0[px[:, 0].astype(np.int), px[:, 1].astype(np.int)] = depth
        pts0 = self.sample_randomly(c_pc)

        # get second depth map
        pc_dir = '%s/sequences/%02d/velodyne/%06d.bin' \
                   % (self.root_dir, self.dicts[idx]['sequence_id'], self.dicts[idx]['image_id']+1)
        pc = np.fromfile(pc_dir, dtype=np.float32, count=-1).reshape([-1, 4])
        c_pc = np.dot(pc[:, 0:3], calib['Tr'][0:3, 0:3].T) + calib['Tr'][0:3, 3] + calib['d2'].T
        c_pc = c_pc[c_pc[:, 2] > 1]
        c_pc = c_pc[c_pc[:, 2] < 80]

        px = np.dot(c_pc, calib['K2'].T)
        depth = px[:, 2]
        px[:, 0] = np.round(px[:, 0] / depth)
        px[:, 1] = np.round(px[:, 1] / depth)

        ind = (px[:, 0] < w) & (px[:, 0] >= 0) &\
              (px[:, 1] < h) & (px[:, 0] >= 0)
        c_pc = c_pc[ind]
        depth = depth[ind]
        px = px[ind]

        trail = np.argsort(depth, -1, 'quicksort')[::-1]
        depth = depth[trail]
        c_pc = c_pc[trail]
        px = px[trail, :]
        depthmap1 = np.zeros(image_size)
        depthmap1[px[:, 0].astype(np.int), px[:, 1].astype(np.int)] = depth
        pts1 = self.sample_randomly(c_pc)

        return depthmap0.astype(np.float32).transpose(), \
               depthmap1.astype(np.float32).transpose(), \
               pts0.astype(np.float32).transpose(), \
               pts1.astype(np.float32).transpose()

    def get_calib(self, idx):
        calib_dir = '%s/sequences/%02d/calib.txt' \
                    % (self.root_dir, self.dicts[idx]['sequence_id'])
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(calib_dir, 'r') as f:
            for i in range(5):
                line = f.readline()
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' ')))).reshape(3, 4)
                    except ValueError:
                        pass  # casting error: data[key] already eq. value, so pass

        delta_z = data['P2'][2, 3]
        delta_x = (data['P2'][0, 3] - data['P2'][0, 2] * delta_z) / data['P2'][0, 0]
        delta_y = (data['P2'][1, 3] - data['P2'][1, 2] * delta_z) / data['P2'][1, 1]
        data['K2'] = data['P2'][0:3, 0:3]
        data['d2'] = np.array([delta_x, delta_y, delta_z]).reshape(3, 1)
        return data

    def __len__(self):
        return len(self.dicts)

    def __getitem__(self, idx):
        first_image, second_image, bbox = self.get_image_pair(idx)
        calib = self.get_calib(idx)
        calib['K2'][0, 2] = calib['K2'][0, 2] - bbox[0]
        calib['K2'][1, 2] = calib['K2'][1, 2] - bbox[1]
        first_depth, second_depth, first_pts, second_pts = self.get_depth_map(idx, calib, first_image.size)
        second_di = di.nearest_filling(second_depth.copy())
        K = calib['K2'].astype(np.float32)
        sh = float(self.im_size[0] / 352)
        sw = float(self.im_size[1] / 1216)
        K[0, 0] *= sw
        K[0, 2] *= sw
        K[1, 1] *= sh
        K[1, 2] *= sh

        # from PIL import ImageDraw
        # image_a = transforms.ToPILImage()(first_image.squeeze(0)).convert('RGB')
        # image_b = transforms.ToPILImage()(second_image.squeeze(0)).convert('RGB')
        # draw = ImageDraw.Draw(image_b)
        # for num in range(7000):
        #     draw.point((second_ind[num, 0], second_ind[num, 1]))
        # image_b.save("%03d_b.png" % idx)
        # image_a.save("%03d_a.png" % idx)

        pose = self.get_pose(idx).astype(np.float32)
        sample = {'first_im': self.im_trans(first_image),
                  'second_im': self.im_trans(second_image),
                  'first_dm': self.dm_trans(first_depth),
                  'second_dm': self.dm_trans(second_depth),
                  'second_di': self.dm_trans(second_di),
                  'first_pts': first_pts,
                  'second_pts': second_pts,
                  'camera': np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32),
                  'pose': pose}

        return sample


if __name__ == '__main__':
    root_dir = '/data/szb/kitti_odom'
    data_dir = 'filenames/00.txt'
    import cv2
    data_set = KittiOdomLoader(root_dir, data_dir, mode='train', im_size=(352, 1216), pt_size=1024*8)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)

    # print('dataset num is ', len(data_loader))
    # pose = np.array([[9.9997503e-01, 2.8443412e-04, 7.0515764e-03, 2.9644728e-02],
    #                  [-2.8766567e-04, 9.9999982e-01, 4.5738302e-04, 1.5263738e-02],
    #                  [-7.0513533e-03, -4.5940082e-04, 9.9997497e-01, 9.0525538e-01],
    #                  [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
    #
    # """Check if a matrix is a valid transformation matrix."""
    # bottom_row = np.append(np.zeros(4 - 1), 1.)
    #
    # a1 = pose.shape == (4, 4)
    # a2 = np.array_equal(pose[4 - 1, :], bottom_row)
    # rot = pose[0:3, 0:3]
    # a3 = rot.shape == (3, 3)
    # a4 = np.isclose(np.linalg.det(rot), 1.)
    # rot_id = rot.T.dot(rot)
    # a5 = np.allclose(rot.T.dot(rot), np.identity(3))
    # se3_tr = SE3.from_matrix(pose)
    for data in data_loader:
        ref_ptc = data['second_pts']
        cam = data['camera']
        b, _, n = ref_ptc.shape
        K = torch.eye(3, dtype=cam.dtype, device=cam.device).view(1, 3, 3).repeat(b, 1, 1)
        K[:, 0, 0] = cam[:, 0]
        K[:, 1, 1] = cam[:, 1]
        K[:, 0, 2] = cam[:, 2]
        K[:, 1, 2] = cam[:, 3]
        pix = torch.bmm(K, ref_ptc)
        pix[:, 0, :] = pix[:, 0, :] / pix[:, 2, :]
        pix[:, 1, :] = pix[:, 1, :] / pix[:, 2, :]
        px = pix[:, :2, :].squeeze().data.cpu().numpy()

        # second_pts = data['second_pts'].squeeze().data.cpu().numpy()
        # cam = data['camera'].squeeze().data.cpu().numpy()
        # K = np.eye(3)
        # K[0, 0] = cam[0]
        # K[1, 1] = cam[1]
        # K[0, 2] = cam[2]
        # K[1, 2] = cam[3]
        # px = np.round(np.dot(K, second_pts))
        # depth = px[2, :]
        # px[0, :] = px[0, :] / depth
        # px[1, :] = px[1, :] / depth
        ind_map = np.zeros((352, 1216), np.uint8)
        ind_map[px[1, :].astype(np.int), px[0, :].astype(np.int)] = 255
        cv2.imwrite('distribution.png', ind_map)
        # print('im size:', im.size())
        # print('gt size:', gt.size())

        # print(gt)
        # print(torch.max(gt))
        # print(torch.min(gt))

        # print(im)
        #
        # if i == 0:
        #     break
