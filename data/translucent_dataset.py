import glob
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
import scipy.ndimage as ndimage


class TranslucentDataset(BaseDataset):
    """
    todo:
        - thickness map?
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        if is_train:
            pass
        else:
            parser.set_defaults(num_test=18000)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """

        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.isTwoshot = opt.isTwoshot
        self.isEdit = opt.isEdit
        # get scene properties
        csv_path = os.path.join(opt.dataroot, opt.phase + '.csv')
        self.csv = pd.read_csv(csv_path,
                               sep=';',
                               header=None,
                               index_col=0,
                               names=['obj', 'envmap', 'roughness',
                                      '_', 'normal', 'radiance',
                                      'sigma_t', 'albedo', 'g',
                                      'surface_opacity', 'obj_scale',
                                      'obj_rotate_x', 'obj_rotate_y', 'obj_rotate_z',
                                      'obj_translate_x', 'obj_translate_y', 'obj_translate_z',
                                      'env_rotate_x', 'env_rotate_y', 'env_rotate_z',
                                      'uv_scale'])
        csv_edit_path = os.path.join(opt.dataroot, opt.phase + '_edit.csv')
        self.csv_edit = pd.read_csv(csv_edit_path,
                                    sep=';',
                                    header=None,
                                    index_col=0,
                                    names=['sigma_t', 'albedo', 'g'])

        # get the image paths of your dataset;

        '''if opt.phase == 'train':
            scene_list = np.load(os.path.join(opt.dataroot, 'scene_list.npy'))
            self.image_paths = [os.path.join(opt.dataroot, opt.phase, str(num)) for num in scene_list]
        else:'''
        self.image_paths = sorted(glob.glob(os.path.join(opt.dataroot, opt.phase, '[0-9]*')))

        self.transform_rgb = get_transform(grayscale=False)
        self.transform_gray = get_transform(grayscale=True)

    def __getitem__(self, index):

        base_path = self.image_paths[index]
        num = int(base_path.split(os.sep)[-1])

        sigma_t = self.csv['sigma_t'][num]
        sigma_t = torch.tensor([float(i) for i in sigma_t.split(',')], dtype=torch.float32)

        albedo = self.csv['albedo'][num]
        albedo = torch.tensor([float(i) for i in albedo.split(',')], dtype=torch.float32)

        g = self.csv['g'][num]
        g = torch.tensor(g, dtype=torch.float32)

        radiance = self.csv['radiance'][num]
        radiance = torch.tensor(radiance, dtype=torch.float32) / 100

        normal_path = os.path.join(base_path, 'normal.png')
        normal_image = self.load_image(normal_path, mode='RGB')

        rough_path = os.path.join(base_path, 'roughness.png')
        rough_image = self.load_image(rough_path, mode='L')

        depth_path = os.path.join(base_path, 'depth.npy')
        depth_image = self.load_npy(depth_path)

        scene_path = os.path.join(base_path, 'scene.png')
        scene_image = self.load_image(scene_path, mode="RGB")

        mask_path = os.path.join(base_path, 'mask.png')
        mask_image = self.load_mask(mask_path)

        coeffs_path = os.path.join(base_path, 'coeffs.npy')
        coeffs = np.load(coeffs_path).transpose([1, 0])[:, 0:9].astype(np.float32)

        res = {'scene': scene_image, 'normal': normal_image, 'depth': depth_image, 'rough': rough_image, 'mask': mask_image,
               'g': g, 'sigma_t': sigma_t, 'albedo': albedo,
               'coeffs': coeffs, 'radiance': radiance}

        if self.isTwoshot:
            non_flash_path = os.path.join(base_path, 'non_flash.png')
            non_flash_image = self.load_image(non_flash_path)
            res['non_flash'] = non_flash_image

        if self.isEdit:
            albedo_edit_path = os.path.join(base_path, 'albedo.png')
            albedo_edit_image = self.load_image(albedo_edit_path)
            res['albedo_edit'] = albedo_edit_image

            sigma_t_edit_path = os.path.join(base_path, 'sigma_t.png')
            sigma_t_edit_image = self.load_image(sigma_t_edit_path)
            res['sigma_t_edit'] = sigma_t_edit_image

            g_edit_path = os.path.join(base_path, 'g.png')
            g_edit_image = self.load_image(g_edit_path)
            res['g_edit'] = g_edit_image

            new_sigma_t = self.csv_edit['sigma_t'][num]
            new_sigma_t = torch.tensor([float(i) for i in new_sigma_t.split(',')], dtype=torch.float32)
            res['new_sigma_t'] = new_sigma_t

            new_albedo = self.csv_edit['albedo'][num]
            new_albedo = torch.tensor([float(i) for i in new_albedo.split(',')], dtype=torch.float32)
            res['new_albedo'] = new_albedo

            new_g = self.csv_edit['g'][num]
            new_g = torch.tensor(new_g, dtype=torch.float32)
            res['new_g'] = new_g

        if self.phase == 'test':
            res['num'] = num
            res['image_paths'] = base_path
        return res

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def load_image(self, path, mode='RGB'):
        """Load and transform image based on the given path and mode."""
        image = Image.open(path).convert(mode)
        assert image.size[0] == 256
        if mode == "RGB":
            data = self.transform_rgb(image)
        if mode == "L":
            data = self.transform_gray(image)
        return data

    def load_mask(self, path):
        mask = Image.open(path).convert('L')
        assert mask.size[0] == 256
        mask = np.asarray(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.999999).astype(dtype=np.int)
        mask = ndimage.binary_erosion(mask, structure=np.ones((2, 2)))
        mask = mask[np.newaxis, :, :]
        return mask

    def load_npy(self, path):
        data = torch.tensor(np.load(path), dtype=torch.float32)
        return data