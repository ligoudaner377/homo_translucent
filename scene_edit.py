import os
import numpy as np
import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import random
import json



def save_image(img, img_name, oup_dir):
    img = img[0, :, :, :]
    img = (img * 255.0).cpu().numpy().astype(np.uint8)
    if img.shape[0] == 1:
        img = img[0, :, :]
        Image.fromarray(img, 'L').save(oup_dir+'/{}.png'.format(img_name))
    else:
        img = np.transpose(img, (1, 2, 0))
        Image.fromarray(img, 'RGB').save(oup_dir+'/{}.png'.format(img_name))

def generate_random_scatter_para():
    scatter_dict = {}

    albedo_0 = [random.uniform(0.3, 0.95), random.uniform(0.3, 0.95), random.uniform(0.3, 0.95)]
    albedo_1 = [random.uniform(0.3, 0.95), random.uniform(0.3, 0.95), random.uniform(0.3, 0.95)]
    albedo_2 = [random.uniform(0.3, 0.95), random.uniform(0.3, 0.95), random.uniform(0.3, 0.95)]
    r_list = list(np.linspace(albedo_0[0], albedo_1[0], 50)) + list(np.linspace(albedo_1[0], albedo_2[0], 50)) + list(np.linspace(albedo_2[0], albedo_0[0], 50))
    g_list = list(np.linspace(albedo_0[1], albedo_1[1], 50)) + list(np.linspace(albedo_1[1], albedo_2[1], 50)) + list(np.linspace(albedo_2[1], albedo_0[1], 50))
    b_list = list(np.linspace(albedo_0[2], albedo_1[2], 50)) + list(np.linspace(albedo_1[2], albedo_2[2], 50)) + list(np.linspace(albedo_2[2], albedo_0[2], 50))
    albedo_list = [(r_list[i], g_list[i], b_list[i]) for i in range(150)]

    sigma_t_list = list(np.linspace(0.2, 0.8, 50))

    g_list = list(np.linspace(0.01, 0.89, 50))

    scatter_dict['albedo'] = albedo_list
    scatter_dict['sigma_t'] = sigma_t_list
    scatter_dict['g'] = g_list
    return scatter_dict


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    edit_list = [4414, 1349, 6397, 2861, 11270, 8209, 11929,
                 13689, 2287, 2483, 1039, 474, 5341, 1064, 10750]
    for data in dataset:
        num = data['num'].item()
        if (num in edit_list) or opt.dataset_mode == 'real':
            print('editing num: {}'.format(num))
            if opt.dataset_mode == 'translucent':
                with open('/home2/lch/project/render_data3/scene_edit/{}.json'.format(num), 'r') as f:
                    scatter_dict = json.load(f)
            else:
                scatter_dict = generate_random_scatter_para()

            oup_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch, 'scene_edit', str(num))
            if not os.path.exists(oup_dir):
                os.makedirs(oup_dir)

            with torch.no_grad():
                model.set_input(data)
                model.parameter_estimating()
                model.direct_rendering()
                # record original estimated parameter
                g_origin = model.scatter_predict[:, 0:1]
                sigma_t_origin = model.scatter_predict[:, 1:4]
                albedo_origin = model.scatter_predict[:, 4:7]
                radiance_origin = model.scatter_predict[:, 7:8]
                for i, albedo in enumerate(scatter_dict['albedo']):
                    model.set_input(data)
                    model.new_albedo = model.normalize_albedo(torch.tensor(albedo, dtype=torch.float32).expand_as(albedo_origin)).cuda()
                    model.parameter_estimating()
                    model.direct_rendering()
                    model.neural_rendering(mode='albedo')
                    model.compute_visuals()
                    save_image(model.albedo_edit_predict/2 + 0.5, 'albedo_{}'.format(i), oup_dir)

                '''for i, sigma_t in enumerate(scatter_dict['sigma_t']):
                    model.set_input(data)
                    model.new_sigma_t = model.normalize(torch.tensor(sigma_t, dtype=torch.float32).expand_as(sigma_t_origin)).to('cuda:0')
                    model.parameter_estimating()
                    model.direct_rendering()
                    model.neural_rendering(mode='sigma_t')
                    model.compute_visuals()
                    save_image(model.sigma_t_edit_predict/2 + 0.5, 'sigma_t_{}'.format(i), oup_dir)

                for i, g in enumerate(scatter_dict['g']):
                    model.set_input(data)
                    model.new_g = model.normalize_g(torch.tensor(g, dtype=torch.float32).expand_as(g_origin)).to('cuda:0')
                    model.parameter_estimating()
                    model.direct_rendering()
                    model.neural_rendering(mode='g')
                    model.compute_visuals()
                    save_image(model.g_edit_predict / 2 + 0.5, 'g_{}'.format(i), oup_dir)'''














