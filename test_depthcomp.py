import argparse
import os
import time

import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import cv2

from data.depth_loader import KittiDepthLoader
from models.nets import FusionComp
from logger.metrics import AverageMeter, Result
from logger.vis_utils import save_depth_as_uint16png
import utils

parser = argparse.ArgumentParser(description='Training the network')
parser.add_argument('--root_dir', type=str, default='/home/song/Documents/data/kitti_depth',
                    help='path to the dataset folder./data/szb/kitti_depth /home/song/Documents/data/kitti_depth'
                    )
parser.add_argument('--val_data_dir', type=str, default='data/filenames/depth_vals.txt',
                    help='path to the validation dataset file. '
                    )
parser.add_argument('--test_data_dir', type=str, default='data/filenames/depth_test.txt',
                    help='path to the test dataset file.'
                    )
parser.add_argument('--output_directory', type=str, default='results',
                    help='where save dispairities for tested images'
                    )
parser.add_argument('--resume',type=str,
                    default='results/depthcomp_model.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--mode', type=str, default='val', choices=('test', 'val'),
                    help='mode: train or test (default: train)'
                    )
parser.add_argument('--image_size', type=tuple, default=(352, 1216),
                    help='image size in the network, as type tuple (height, width)'
                    )
args = parser.parse_args()


def rgb_to_gray(rgb_tensor):
    B, _, H, W = rgb_tensor.shape
    gray = (rgb_tensor[:, 0] * 0.299 + rgb_tensor[:, 1] * 0.587 + rgb_tensor[:, 2] * 0.114).view(B, 1, H, W)
    return gray

cuda = torch.cuda.is_available()

if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

checkpoint = None
if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}' ... ".format(args.resume),
          end='')
    checkpoint = torch.load(args.resume, map_location=device)
    print("Completed.")
else:
    print("No checkpoint found at '{}'".format(args.resume))

model = FusionComp().to(device)

if checkpoint is not None:
    model.load_state_dict(checkpoint['model'])
    print("=> checkpoint state loaded.")
model.eval()

model = torch.nn.DataParallel(model)

# Data loading code
print("=> creating data loaders ... ")

if 'test' in args.mode:
    dataset = KittiDepthLoader(args.root_dir, args.test_data_dir, 'test', args.image_size)
elif 'val' in args.mode:
    dataset = KittiDepthLoader(args.root_dir, args.val_data_dir, 'val', args.image_size)

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=2,
                                     pin_memory=True,)

# main loop
print("=> starting main loop ...")

average_meter = AverageMeter()

save_folder = os.path.join(args.output_directory,
                           args.mode + "_output")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

rmse_list = list()
show_results = False
cmap = plt.cm.jet

for i, batch_data in enumerate(loader):
    start = time.time()
    batch_data = utils.to_device(batch_data, device)
    # we use log depth to represent the depth values
    in_disp = torch.where(batch_data['d'] == 0.,
                          torch.zeros_like(batch_data['d']),
                          torch.reciprocal(batch_data['d']))
    rgb = batch_data['rgb']
    gray = rgb_to_gray(rgb)
    data_time = time.time() - start

    start = time.time()
    with torch.no_grad():
        pred_depth = torch.reciprocal(0.49 * model(rgb, in_disp) + 0.01)

    gpu_time = time.time() - start

    if show_results:
        x = pred_depth
        fea = np.squeeze(x.data.cpu().numpy())
        fea = (fea - np.min(fea)) / (np.max(fea) - np.min(fea))
        fea = 255 * cmap(fea)[:, :, :3]  # H, W, C
        rgb = 255 * np.squeeze(rgb.transpose(2, 3).data.cpu().numpy()).transpose()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        cv2.imshow('image', bgr.astype('uint8'))
        cv2.imshow('depth', fea.astype('uint8'))
        cv2.waitKey(5)

    # measure accuracy and record loss
    if args.mode == 'val':
        gt_depth = batch_data['gt']
        result = Result()
        pred_depth_cpu = pred_depth.data.cpu().numpy().squeeze()
        pred_depth_cpu = cv2.bilateralFilter(cv2.medianBlur(pred_depth_cpu, 5), 5, 1.5, 2.0)
        pred_depth = torch.from_numpy(pred_depth_cpu).cuda().unsqueeze(0).unsqueeze(0)
        result.evaluate(pred_depth.data, gt_depth.data)
        average_meter.update(result, gpu_time, data_time)
        rmse_list.append([i, result.rmse])

        print('\t ==> frame: ', i, 'rmse', result.rmse, 'mae', result.mae, 'iMae', result.imae, 'finished.')
    elif args.mode == 'test':
        img = torch.squeeze(pred_depth.data.cpu()).numpy()
        img = cv2.bilateralFilter(cv2.medianBlur(img, 5), 5, 1.5, 2.0)
        filename = os.path.join(save_folder, '{0:010d}.png'.format(i))
        save_depth_as_uint16png(img, filename)
        print('\t ==>', filename, '\t saved.')


if args.mode == 'val':
    avg = average_meter.average()
    print(''
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Photo={average.photometric:.3f}\n'
          'iRMSE={average.irmse:.3f}\n'
          'iMAE={average.imae:.3f}\n'
          'squared_rel={average.squared_rel}\n'
          'silog={average.silog}\n'
          'Delta1={average.delta1:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}'.format(average=avg, time=avg.gpu_time))

    with open(os.path.join(save_folder, 'sparse_pnp.txt'), 'a') as txtfile:
        txtfile.write(
            ("\nrmse={:.3f}\n" +
             "mae={:.3f}\n" + "silog={:.3f}\n" + "squared_rel={:.3f}\n" +
             "irmse={:.3f}\n" + "imae={:.3f}\n" + "mse={:.3f}\n" +
             "absrel={:.3f}\n" + "lg10={:.3f}\n" + "delta1={:.3f}\n" +
             "t_gpu={:.4f}\n").format(avg.rmse, avg.mae, avg.silog,
                                    avg.squared_rel, avg.irmse,
                                    avg.imae, avg.mse, avg.absrel,
                                    avg.lg10, avg.delta1,
                                    avg.gpu_time/len(loader)))

    with open(os.path.join(save_folder, 'ip_val_results_sort.txt'), 'w') as txtfile:
        for i in range(len(rmse_list)):
            txtfile.write("%d %.3f\n" % (rmse_list[i][0], rmse_list[i][1]))
