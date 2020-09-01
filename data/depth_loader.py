from PIL import Image
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import data.depth_interpolate as di


class KittiDepthLoader(Dataset):
    def __init__(self, root_dir, data_dir, mode, im_size, pose_src=None):
        self.im_size = im_size
        self.mode = mode
        self.pose_src = pose_src

        rgb_list = list()
        d_list = list()
        gt_list = list()
        k_list = list()

        #transforms
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
             transforms.ToTensor(), ]
        )

        # data list
        with open(data_dir, "r") as f:
            for line in f:
                line = line.strip("\n")
                fields = line.split(" ")
                rgb_list.append('%s/%s' % (root_dir, fields[0]))
                d_list.append('%s/%s' % (root_dir, fields[1]))
                k_list.append('%s/%s' % (root_dir, fields[2]))
                if mode != 'test':
                    gt_list.append('%s/%s' % (root_dir, fields[3]))

        if pose_src is not None:
            if 'vo' in pose_src:
                pose_dir = 'data/filenames/depth_vo_pose.txt'
            elif 'pnp' in pose_src:
                pose_dir = 'data/filenames/depth_pnp_pose.txt'
            else:
                raise NotImplementedError
            self.pose_list = list()
            with open(pose_dir, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam2 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam2 = T_w_cam2.reshape(3, 4)
                    T_w_cam2 = np.vstack((T_w_cam2, [0, 0, 0, 1]))
                    self.pose_list.append(T_w_cam2)
        self.dicts = {'rgb': rgb_list, 'd': d_list, 'k': k_list, 'gt': gt_list}

    def get_image(self, idx):
        image = Image.open(self.dicts['rgb'][idx])
        width = image.size[0]
        height = image.size[1]
        delta_w = width - 1216
        delta_h = height - 352
        left_d = int(delta_w / 2)
        right_d = width - (delta_w - left_d)
        image = image.crop([left_d, delta_h, right_d, height])
        return image, [left_d, delta_h, right_d, height]

    def get_depth_map(self, idx, crop_box):
        depth = Image.open(self.dicts['d'][idx])
        depth = depth.crop(crop_box)
        depth = np.array(depth, dtype=np.float32) / 256.
        return depth

    def get_gt_map(self, idx, crop_box):
        depth = Image.open(self.dicts['gt'][idx])
        depth = depth.crop(crop_box)
        depth = np.array(depth, dtype=np.float32) / 256.
        return depth

    def get_calib(self, idx, crop_box):
        calib = open(self.dicts['k'][idx], "r")
        lines = calib.readlines()
        if len(lines) > 5:
            P_rect_line = lines[25]
            Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
            Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                              (3, 4)).astype(np.float32)
            K = Proj[:3, :3]  # camera matrix
        else:
            P_rect_line = lines[0]
            Proj_str = P_rect_line.split(" ")[:-1]
            Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                              (3, 3)).astype(np.float32)
            K = Proj[:3, :3]  # camera matrix
        K[0, 2] -= crop_box[0]
        K[1, 2] -= crop_box[1]
        return K

    def get_near_idx(self, idx):
        # idx_near = [idx+1, idx-1]
        # idx_near = [idx for idx in idx_near if (idx >= 0) and (idx < self.__len__())]
        # if len(idx_near) == 1:
        #     return idx_near*2
        # else:
        #     rgb_dir = self.dicts['rgb'][idx].split('/')[:-1]
        #     rgb_near_dir = self.dicts['rgb'][idx_near[0]].split('/')[:-1]
        #     if ('/'.join(rgb_dir) != '/'.join(rgb_near_dir)):
        #         idx_near[0] = idx_near[1]
        #     rgb_near_dir = self.dicts['rgb'][idx_near[1]].split('/')[:-1]
        #     if ('/'.join(rgb_dir) != '/'.join(rgb_near_dir)):
        #         idx_near[1] = idx_near[0]
        #     return idx_near

        candidates = [idx+1, idx+2, idx+3, idx-1, idx-2, idx-3]
        candidates = [idx_near for idx_near in candidates if (idx_near>=0) and (idx_near<self.__len__())]
        idx_near = idx
        if len(candidates) > 0:
            for i in range(10):
                idx_near = random.sample(candidates, 2)
                rgb_dir = self.dicts['rgb'][idx].split('/')[:-1]
                rgb_near_dir0 = self.dicts['rgb'][idx_near[0]].split('/')[:-1]
                rgb_near_dir1 = self.dicts['rgb'][idx_near[1]].split('/')[:-1]
                if ('/'.join(rgb_dir) == '/'.join(rgb_near_dir0)) and \
                        ('/'.join(rgb_dir) == '/'.join(rgb_near_dir1)):
                    break
        return idx_near

    def __len__(self):
        return len(self.dicts['rgb'])

    def __getitem__(self, idx):
        sh = float(self.im_size[0] / 352)
        sw = float(self.im_size[1] / 1216)
        image, crop_box = self.get_image(idx)
        depth = self.get_depth_map(idx, crop_box)

        # remove occlusion
        min_depth = di.dilate_interpolate(depth) * (depth > 0.1).astype(depth.dtype)
        mask = min_depth < depth*0.95
        depth[mask] = 0.

        # sparsity invariant
        # valid_mask = depth > 0.1
        # drop_mask = np.random.binomial(1, 8192./valid_mask.sum(), depth.shape)
        # mask = (valid_mask & drop_mask).astype(np.float32)

        # cv2.imshow('no_gt', depth)
        # cv2.waitKey(0)
        # mask = np.random.binomial(1, prob_keep, depth.shape)

        # gt_depth = self.get_gt_map(idx, crop_box)
        # mask_sparse = (gt_depth > 0) & (depth > 0)
        # error = mask_sparse.astype(np.float) * np.abs(gt_depth - depth)
        # mask = ((depth > 0) & (error < 0.1)).astype(np.float32)

        depth_in = di.fill_in_fast(depth.copy(), extrapolate=True)
        # depth_in = di.nearest_filling(depth.copy())
        # depth_in = di.interpolate_depth(depth, depth>0.1)

        K = self.get_calib(idx, crop_box)

        sample = {'rgb': self.im_trans(image),
                  'd': self.dm_trans(depth),
                  'di': self.dm_trans(depth_in),
                  'K': np.array([K[0, 0]*sw, K[1, 1]*sh, K[0, 2]*sw, K[1, 2]*sh], dtype=np.float32),
                  }

        if self.mode != 'test':
            gt_depth = self.get_gt_map(idx, crop_box)
            sample['gt'] = self.dm_trans(gt_depth)

        # depth_in_log = np.where(depth_in==0, 0., np.reciprocal(depth_in))
        # depth_gt_log = np.where(gt_depth==0, 0., np.reciprocal(gt_depth))
        # error = np.where(depth_in - gt_depth==0, 0., np.reciprocal(depth_in-gt_depth))
        # mask = gt_depth > 0.1
        # error = error[mask]
        # # error = error[error>0.1]
        # import matplotlib.pyplot as plt
        # plt.hist(error)
        # plt.show()

        if self.pose_src is not None:
            # adjacent idx data
            idx_near1, idx_near2 = self.get_near_idx(idx)
            image_near1, crop_box_near1 = self.get_image(idx_near1)
            image_near2, crop_box_near2 = self.get_image(idx_near2)

            # pose
            pose1 = np.dot(np.linalg.inv(self.pose_list[idx_near1]), self.pose_list[idx])
            pose2 = np.dot(np.linalg.inv(self.pose_list[idx_near2]), self.pose_list[idx])
            sample['rgb_near1'] = self.im_trans(image_near1)
            sample['rgb_near2'] = self.im_trans(image_near2)
            sample['R1'] = pose1[:3, :3].astype(np.float32)
            sample['t1'] = pose1[:3, 3:].astype(np.float32)
            sample['R2'] = pose2[:3, :3].astype(np.float32)
            sample['t2'] = pose2[:3, 3:].astype(np.float32)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    root_dir = '/home/song/Documents/data/kitti_depth'
    data_dir = 'filenames/depth_vals.txt'
    data_set = KittiDepthLoader(root_dir, data_dir, mode='val', im_size=(352, 1216))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)
    cmap = plt.cm.jet

    for data in data_loader:
        gt = data['gt']
        d = data['di']
        depth = np.squeeze(d.data.cpu().numpy())
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        depth = 255 * cmap(depth)[:, :, :3]

        gt = np.squeeze(gt.data.cpu().numpy())
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        gt = 255 * cmap(gt)[:, :, :3]

        cv2.imwrite('../debug/depth_i.png', depth.astype('uint8'))
        cv2.imwrite('../debug/depth_gt.png', gt.astype('uint8'))
        break

