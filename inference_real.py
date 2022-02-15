import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image

def save_image(img, img_name, num, oup_dir):
    if not os.path.exists(os.path.join(oup_dir, num)):
        os.makedirs(os.path.join(oup_dir, num))
    img = img[0, :, :, :] * 0.5 + 0.5
    img = (img * 255.0).cpu().numpy().astype(np.uint8)
    if img.shape[0]==1:
        img = img[0, :, :]
        Image.fromarray(img, 'L').save(os.path.join(oup_dir, num, '{}.png'.format(img_name)))
    else:
        img = np.transpose(img, (1, 2, 0))
        Image.fromarray(img, 'RGB').save(os.path.join(oup_dir, num, '{}.png'.format(img_name)))

def save_scatter_para(para, num, oup_dir):
    if not os.path.exists(os.path.join(oup_dir, num)):
        os.makedirs(os.path.join(oup_dir, num))
    para = (para[0, :] * 0.5 + 0.5).cpu().numpy()
    np.save(os.path.join(oup_dir, num, 'scatter_para.npy'), para)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    #opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    oup_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch, 'real')
    if not os.path.exists(oup_dir):
        os.makedirs(oup_dir)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    img_list = ['normal_predict',
                'albedo_predict',
                'rough_predict',
                'depth_predict_vis',
                'direct_predict',
                'scene_predict']
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()
        model.compute_visuals()           # run inference

        for img_name in img_list:
            if hasattr(model, img_name):
                img = getattr(model, img_name)
                save_image(img, img_name, str(data['num'].item()), oup_dir)
        save_scatter_para(model.scatter_predict, str(data['num'].item()), oup_dir)








