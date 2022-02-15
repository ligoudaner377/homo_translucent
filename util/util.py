"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import json

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(model, num, opt):
    out_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch, 'data')
    mkdir(out_dir)
    out_file = os.path.join(out_dir, '{}.json'.format(num.item()))

    coeffs_predict = model.coeffs_predict.squeeze().cpu().numpy()
    scatter_predict = model.scatter_predict.squeeze().cpu().numpy()
    g_predict = model.inverse_normalize_g(scatter_predict[0])
    radiance_predict = model.inverse_normalize_radiance(scatter_predict[-1])
    sigma_t_predict = model.inverse_normalize(scatter_predict[1:4])
    albedo_predict = model.inverse_normalize_albedo(scatter_predict[4:7])

    coeffs_para = model.coeffs_para.squeeze().cpu().numpy()
    scatter_para = model.scatter_para.squeeze().cpu().numpy()
    g_para = model.inverse_normalize_g(scatter_para[0])
    radiance_para = model.inverse_normalize_radiance(scatter_para[-1])
    sigma_t_para = model.inverse_normalize(scatter_para[1:4])
    albedo_para = model.inverse_normalize_albedo(scatter_para[4:7])

    data = dict()
    data['coeffs_l1'] = float(np.mean(np.abs(coeffs_predict - coeffs_para)))
    data['radiance_l1'] = float(np.abs(radiance_predict - radiance_para))
    data['sigma_t_l1'] = float(np.mean(np.abs(sigma_t_predict - sigma_t_para)))
    data['albedo_l1'] = float(np.mean(np.abs(albedo_predict - albedo_para)))
    data['g_l1'] = float(np.abs(g_predict - g_para))

    data['g_para'] = float(g_para)
    data['albedo_para'] = albedo_para.tolist()
    data['sigma_t_para'] = sigma_t_para.tolist()
    data['coeffs_para'] = coeffs_para.tolist()

    data['g_predict'] = float(g_predict)
    data['albedo_predict'] = albedo_predict.tolist()
    data['sigma_t_predict'] = sigma_t_predict.tolist()
    data['coeffs_predict'] = coeffs_predict.tolist()

    normal_image = model.inverse_normalize(model.normal_image)
    normal_predict = model.inverse_normalize(model.normal_predict)
    normal_l1 = model.imageLoss(normal_image, normal_predict)
    data['normal_l1'] = normal_l1.item()

    rough_image = model.inverse_normalize(model.rough_image)
    rough_predict = model.inverse_normalize(model.rough_predict)
    rough_l1 = model.imageLoss(rough_image, rough_predict)
    data['rough_l1'] = rough_l1.item()

    depth_image = model.inverse_normalize(model.depth_image_vis)
    depth_predict = model.inverse_normalize(model.depth_predict_vis)
    depth_l1 = model.imageLoss(depth_image, depth_predict)
    data['depth_l1'] = depth_l1.item()

    if hasattr(model, 'scene_predict'):
        scene_image = model.inverse_normalize(model.scene_image)
        scene_predict = model.inverse_normalize(model.scene_predict)
        scene_l1 = model.imageLoss(scene_image, scene_predict)
        data['scene_l1'] = scene_l1.item()

    if hasattr(model, 'direct_predict'):
        direct_image = model.inverse_normalize(model.direct_image)
        direct_predict = model.inverse_normalize(model.direct_predict)
        direct_l1 = model.imageLoss(direct_image, direct_predict)
        data['direct_l1'] = direct_l1.item()

    # save data
    with open(out_file, 'w') as f:
        json.dump(data, f)



















