import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import numpy as np
from PIL import Image

def save_images(model, num, oup_dir, idx, edit_name):
    scene_predict = model.scene_predict
    scene_predict = torch.squeeze((scene_predict * 0.5 + 0.5) * model.binary_mask).permute(1, 2, 0).cpu().numpy()
    scene_predict = (scene_predict * 255.0).astype(np.uint8)
    if not os.path.exists(os.path.join(oup_dir, str(num))):
        os.makedirs(os.path.join(oup_dir, str(num)))
    Image.fromarray(scene_predict, 'RGB').save(os.path.join(oup_dir, str(num), '{}_{}.png'.format(edit_name, idx)))

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
    oup_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch, 'scene_edit')
    if not os.path.exists(oup_dir):
        os.mkdir(oup_dir)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    edit_list = [7373, 4491, 14748, 9847, 3670, 5040]
    for i, data in enumerate(dataset):
        if data['num'] in edit_list:
            print('editing No.{}'.format(data['num'].item()))
            model.set_input(data)  # unpack data from data loader
            model.test()

            original_surface_opacity = model.new_surface_opacity  # record the original parameter
            original_radiance = model.new_radiance

            ###### fix the radiance and edit surface opacity between [0.55, 0.95] * 2 -1 ####
            for j in range(100):
                new_surface_opacity = torch.tensor((j+1)/100*0.8+0.1, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(model.device)
                model.new_surface_opacity = new_surface_opacity
                # rendering
                with torch.no_grad():
                    model.neural_rendering()
                    # save images
                    save_images(model, data['num'].item(), oup_dir, j, 'surface_opacity')

            model.new_surface_opacity = original_surface_opacity  # restore the original parameter

            ###### fix the surface opacity edit radiance between [0.35, 0.75] * 2 -1 ####
            for j in range(100):
                new_radiance = torch.tensor((j+1)/100*0.8-0.3, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(model.device)
                model.new_radiance = new_radiance
                with torch.no_grad():
                    # rendering
                    model.neural_rendering()
                    # save images
                    save_images(model, data['num'].item(), oup_dir, j, 'radiance')

















