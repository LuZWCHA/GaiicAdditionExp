import copy
import time
import cv2
import kornia as K
import kornia.feature as KF
from kornia.feature.loftr import LoFTR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import glob
import random
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.cm as cm

from kornia_moons.feature import draw_LAF_matches
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from src.loftr import LoFTR, default_cfg
from src.utils.plotting import make_matching_figure

DEVICE = 'cuda:0'
WEIGHT_PATH = 'outdoor_ds.ckpt'

def gaussian_kernel(sigma, sz):
    xpos_vec = np.arange(sz)
    ypos_vec = np.arange(sz)
    output = np.ones([1, 3,sz, sz], dtype=np.single)
    midpos = sz // 2
    for xpos in xpos_vec:
        for ypos in ypos_vec:
            output[:,:,xpos,ypos] = np.exp(-((xpos-midpos)**2 + (ypos-midpos)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return output


def torch_image_translate(input_, tx, ty, interpolation='nearest'):
    # got these parameters from solving the equations for pixel translations
    # on https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
    translation_matrix = torch.zeros([1, 3, 3], dtype=torch.float)
    translation_matrix[:, 0, 0] = 1.0
    translation_matrix[:, 1, 1] = 1.0
    translation_matrix[:, 0, 2] = -2*tx/(input_.size()[2]-1)
    translation_matrix[:, 1, 2] = -2*ty/(input_.size()[3]-1)
    translation_matrix[:, 2, 2] = 1.0
    grid = F.affine_grid(translation_matrix[:, 0:2, :], input_.size()).to(input_.device)
    wrp = F.grid_sample(input_, grid, mode=interpolation)
    return wrp
def Dp(image, xshift, yshift, sigma, patch_size):
    shift_image = torch_image_translate(image, xshift, yshift, interpolation='nearest')#将image在x、y方向移动I`(a)
    diff = torch.sub(image, shift_image)#计算差分图I-I`(a)
    diff_square = torch.mul(diff, diff)#(I-I`(a))^2
    res = torch.conv2d(diff_square, weight =torch.from_numpy(gaussian_kernel(sigma, patch_size)), stride=1, padding=3)#C*(I-I`(a))^2
    return res

def MIND(image, patch_size=7, neigh_size=9, sigma=2, eps=1e-5,image_size0=512,image_size1=512,  name='MIND'):
    # compute the Modality independent neighbourhood descriptor (MIND) of input image.
    # suppose the neighbor size is R, patch size is P.
    # input image is 384 x 256 x input_c_dim
    # output MIND is (384-P-R+2) x (256-P-R+2) x R*R
    
    reduce_size = int((patch_size + neigh_size - 2) / 2)#卷积后减少的size
 
    # estimate the local variance of each pixel within the input image.
    Vimg = torch.add(Dp(image, -1, 0, sigma, patch_size), Dp(image, 1, 0, sigma, patch_size))
    Vimg = torch.add(Vimg, Dp(image, 0, -1, sigma, patch_size))
    Vimg = torch.add(Vimg, Dp(image, 0, 1, sigma, patch_size))#sum(Dp)
    Vimg = torch.div(Vimg,4) + torch.mul(torch.ones_like(Vimg), eps)#防除零
    # estimate the (R*R)-length MIND feature by shifting the input image by R*R times.
    xshift_vec = np.arange( -(neigh_size//2), neigh_size - (neigh_size//2))#邻域计算
    yshift_vec = np.arange(-(neigh_size // 2), neigh_size - (neigh_size // 2))#邻域计算
    iter_pos = 0
    for xshift in xshift_vec:
        for yshift in yshift_vec:
            if (xshift,yshift) == (0,0):
                continue
            MIND_tmp = torch.exp(torch.mul(torch.div(Dp(image, xshift, yshift,  sigma, patch_size), Vimg), -1))#exp(-D(I)/V(I))
            tmp = MIND_tmp[:, :, reduce_size:(image_size0 - reduce_size), reduce_size:(image_size1 - reduce_size)]
            if iter_pos == 0:
                output = tmp
            else:
                output = torch.cat([output,tmp], 1)
            iter_pos = iter_pos + 1
 
    # normalization.
    input_max, input_indexes = torch.max(output, dim=1)
    output = torch.div(output,input_max)
 
    return output


def load_torch_image(fname):
    img = cv2.imread(fname)
    img = K.image_to_tensor(img, False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img


def match(img_path0, img_path1, matcher, device=DEVICE):
    img0 = MIND(load_torch_image(img_path0))
    img1 = MIND(load_torch_image(img_path1))
        
    input_dict = {"image0": K.color.rgb_to_grayscale(img0).to(device), 
                  "image1": K.color.rgb_to_grayscale(img1).to(device)}
    
    with torch.no_grad():
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
        
    return mkpts0, mkpts1
        
def get_F_matrix(mkpts0, mkpts1):

    # Make sure we do not trigger an exception here.
    if len(mkpts0) > 8:
        F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)

        assert F.shape == (3, 3), 'Malformed F?'
    else:
        F = np.zeros((3, 3))

    return F

def match_and_draw(img_path0, img_path1, matcher, device=DEVICE, drop_outliers=False):
    
    img0 = load_torch_image(img_path0)
    img1 = load_torch_image(img_path1)
        
    input_dict = {"image0": K.color.rgb_to_grayscale(img0).to(device), 
                  "image1": K.color.rgb_to_grayscale(img1).to(device)}
    
    with torch.no_grad():
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    if len(mkpts0) > 8:
        F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)

        assert F.shape == (3, 3), 'Malformed F?'
    else:
        F = np.zeros((3, 3))
            
    if drop_outliers:
        mkpts0 = mkpts0[inliers.reshape(-1) > 0]
        mkpts1 = mkpts1[inliers.reshape(-1) > 0]
        inliers = inliers[inliers > 0]

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img0),
        K.tensor_to_image(img1),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None, 
                   'feature_color': (0.2, 0.5, 1), 'vertical': False})
    
    del correspondences, input_dict
    # torch.cuda.empty_cache()


def plot_matching(samples, files):
    for i in range(samples.shape[1]):
        path0 = files[samples[0][i]]
        path1 = files[samples[1][i]]
        print(f'Matching: {path0} to {path1}')
        match_and_draw(path0, path1, matcher)
        plt.show()


def match_and_stitch(img_path0, img_path1, matcher, device=DEVICE, drop_outliers=False, alpha=0.5, show=True, mode="train"):
    img0 = load_torch_image(img_path0)
    img1 = load_torch_image(img_path1)
    if is_night(img0 * 255):
        return None, None
    input_dict = {"image0": K.color.rgb_to_grayscale(img0).to(device), 
                  "image1": K.color.rgb_to_grayscale(img1).to(device)}
    # print(K.color.rgb_to_grayscale(img0).shape, K.color.rgb_to_grayscale(img0).max())
    time_start = time.time_ns()
    
    with torch.no_grad():
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()
    # print(confidence.mean())

    valid_idx = confidence > 0.32
    
    if np.sum(valid_idx) <= 16:
        return None, None

    c = confidence[valid_idx].mean()
    # print(confidence[valid_idx].mean())
    if c < 0.5:
        return None, None
    # print(correspondences)
    mkpts0 = mkpts0[valid_idx]
    mkpts1 = mkpts1[valid_idx]
    if len(mkpts0) > 8:
        F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)

        assert F.shape == (3, 3), 'Malformed F?'
    else:
        F = np.zeros((3, 3))
            
    if drop_outliers:
        mkpts0 = mkpts0[inliers.reshape(-1) > 0]
        mkpts1 = mkpts1[inliers.reshape(-1) > 0]
    stitched_img = stitch_images(img0, img1, mkpts0, mkpts1, alpha)

    if show:
        plt.imshow(cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
    del correspondences, input_dict
    stitched_img =cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    return stitched_img, img1.detach().cpu().numpy().squeeze().transpose((1, 2, 0))

from tps import warp_image
def stitch_images(img0, img1, mkpts0, mkpts1, alpha=0.7):
    
    s = img0.shape[2:]
    img0_np = img0.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
    img1_np = img1.detach().cpu().numpy().squeeze().transpose((1, 2, 0))

    normalized_mkpts0 = copy.deepcopy(mkpts0)
    normalized_mkpts1 = copy.deepcopy(mkpts1)
    dis = np.abs(mkpts0 - mkpts1)
    dis = np.sqrt(dis[:, 0] ** 2 + dis[:, 1] ** 2) < 40
    
    coner0 = np.array(
        [[0,0], [0, 1], [1, 0], [1, 1]]
    )
    
    coner1 = np.array(
        [[0,0], [0, 1], [1, 0], [1, 1]]
    )
    
    normalized_mkpts0 = normalized_mkpts0[dis]
    normalized_mkpts1 = normalized_mkpts1[dis]

    normalized_mkpts0[:, 0] = normalized_mkpts0[:, 0] / s[1]
    normalized_mkpts0[:, 1] = normalized_mkpts0[:, 1] / s[0]
    normalized_mkpts1[:, 0] = normalized_mkpts1[:, 0] / s[1]
    normalized_mkpts1[:, 1] = normalized_mkpts1[:, 1] / s[0]
    
    normalized_mkpts0 = np.concatenate([normalized_mkpts0[:256], ], axis=0)
    normalized_mkpts1 = np.concatenate([normalized_mkpts1[:256], ], axis=0)
    try:
        moved_img0 = warp_image(img0.detach(), normalized_mkpts0, normalized_mkpts1).numpy().squeeze().transpose((1, 2, 0))
    except:
        return img0.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
    print(moved_img0.shape)
    stitched_img = cv2.addWeighted(moved_img0, alpha, img1_np, 1-alpha, 0)
    plt.imshow(stitched_img)
    plt.axis('off')
    plt.show()
    stitched_img = cv2.addWeighted(img0_np, alpha, img1_np, 1-alpha, 0)
    plt.imshow(stitched_img)
    plt.axis('off')
    plt.show()
    return moved_img0


def is_night(image):
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # brightness = hsv_image[:,:,2]
    # avg_brightness = np.mean(brightness)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
    threshold = 0.3
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    total_pixels = gray_image.size
    black_pixels = np.sum(gray_image < 20)
    black_percentage = black_pixels / total_pixels

    return black_percentage >= threshold

import os, tqdm, imageio, shutil
from pathlib import Path

def do_registration(weights, image_pairs, output_dir, show=False):
    if weights is None:
        weights = WEIGHT_PATH
    matcher = LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(WEIGHT_PATH)['state_dict'])
    matcher = matcher.to(DEVICE)
    matcher.eval()

    for img_path0, img_path1 in tqdm.tqdm(image_pairs):
        # img_path0 = f'/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/test/rgb/00429.jpg'
        # img_path1 = f'/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/test/tir/00429.jpg'
        match_and_draw(img_path0, img_path1, matcher)
        plt.show()
        plt.close()
        moved_img0, img1 = match_and_stitch(img_path0, img_path1, matcher, drop_outliers=True,alpha=0.6, show=show, mode='test')
        
        
        tir_img = img1
        img_name = Path(img_path0).name
        rgb_save_path = os.path.join(output_dir, "rgb")
        tir_save_path = os.path.join(output_dir, "tir")
        os.makedirs(os.path.join(rgb_save_path), exist_ok=True)
        os.makedirs(os.path.join(tir_save_path), exist_ok=True)

        rgb_save_path = os.path.join(rgb_save_path, img_name)
        tir_save_path = os.path.join(tir_save_path, img_name)

        # shutil.copy(img_path1, tir_save_path)
        if moved_img0 is not None:
            rgb_img = moved_img0 * 255
            cv2.imwrite(rgb_save_path, rgb_img.astype(np.uint8), )
        else:
            shutil.copy(img_path0, rgb_save_path)
        shutil.copy(img_path1, tir_save_path)

        if show:
            plt.imshow(res)
            plt.show()
            plt.close()

def get_image_pairs(image_root):
    rgb_images = glob.glob(os.path.join(image_root, "rgb/*.jpg"))
    return [(i, i.replace("/rgb/", "/tir/")) for i in rgb_images]


if __name__ == "__main__":
    # do_registration(WEIGHT_PATH, get_image_pairs("/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/train"), "/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/reg_train")
    do_registration(WEIGHT_PATH, get_image_pairs("/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/test"), "/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/reg_test")
    # do_registration(WEIGHT_PATH, get_image_pairs("/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/val"), "/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/reg_val")
