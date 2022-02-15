import glob
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
import scipy.ndimage as ndimage


class RealDataset(BaseDataset):


    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(new_dataset_option=2.0)  # specify dataset-specific default values

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
        # get the image paths of your dataset;
        self.isTwoshot = opt.isTwoshot
        self.image_paths = sorted(glob.glob(os.path.join(opt.dataroot, '[0-9]*')))
        self.transform_rgb = get_transform(grayscale=False)
        self.transform_gray = get_transform(grayscale=True)

    def __getitem__(self, index):
        base_path = self.image_paths[index]
        num = int(base_path.split(os.sep)[-1])

        scene_path = os.path.join(base_path, 'scene.png')
        scene_image = self.load_image(scene_path, mode="RGB")

        mask_path = os.path.join(base_path, 'mask.png')
        mask_image = self.load_mask(mask_path)

        res = {'scene': scene_image,  'mask': mask_image, 'num': num, 'image_paths': base_path}
        if self.isTwoshot:
            non_flash_path = os.path.join(base_path, 'non_flash.png')
            non_flash_image = self.load_image(non_flash_path, mode="RGB")
            res['non_flash'] = non_flash_image
        return res


    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def load_image(self, path, mode='RGB'):
        """Load and transform image based on the given path and mode."""
        image = Image.open(path).convert(mode)
        if image.size[0] != 256:
            image = image.resize((256, 256))
        if mode == "RGB":
            data = self.transform_rgb(image)
        if mode == "L":
            data = self.transform_gray(image)
        return data

    def load_mask(self, path):
        mask = Image.open(path).convert('L')
        if mask.size[0] != 256:
            mask = mask.resize((256, 256))
        mask = np.asarray(mask, dtype=np.float32) / 255.0
        if len(mask.shape) == 3:
            mask = (mask[:, :, 0] > 0.999999).astype(dtype=np.int)
        else:
            mask = (mask > 0.999999).astype(dtype=np.int)
        mask = ndimage.binary_erosion(mask, structure=np.ones((2, 2)))
        mask = mask[np.newaxis, :, :]
        return mask
